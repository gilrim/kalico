import random
from types import SimpleNamespace

import pytest

from klippy.extras import load_cell_v23 as p


@pytest.mark.parametrize(
    "v", [0, 1, 31, 32, 95, 96, -1, -32, -33, 4095, 12288, -(2**31), 144000000]
)
def test_vlq_roundtrip(v):
    data = p.encode_vlq(v)
    assert p.decode_vlq(data, 0) == (v, len(data))


def test_block_roundtrip_and_resync():
    # a real idle frame from the K1-MAX-L_V23 board
    real = bytes.fromhex("06104a97257e")
    assert p.build_block(0, bytes.fromhex("4a")) == real
    payloads, seqs, rest = p.split_blocks(b"\x00\xff" + real + real[:3])
    assert payloads == [bytes.fromhex("4a")] and seqs == [0]
    assert rest == real[:3]


def test_fix_width():
    assert p.fix_width(-10, "%hu") == 65526
    assert p.fix_width(-10, "%hi") == -10
    assert p.fix_width(-1, "%u") == 0xFFFFFFFF


def test_unzip_data():
    # values 100, +5, -300 delta coded; byte lengths 1, 1, 2 -> codes 0, 0, 1
    packed = (
        bytes([3, 0b010000])
        + (100).to_bytes(1, "little", signed=True)
        + (5).to_bytes(1, "little", signed=True)
        + (-300).to_bytes(2, "little", signed=True)
    )
    assert p.unzip_data(packed, True) == [100, 105, -195]


def test_fit_contact_real_probe():
    # fine pass of the first real probe on a K1 Max (z, force counts)
    run = [
        (0.19, -233),
        (0.18, 350),
        (0.17, 3081),
        (0.16, 5630),
        (0.15, 8679),
        (0.14, 11687),
        (0.13, 14839),
        (0.12, 18493),
        (0.11, 21564),
    ]
    z0, stiffness, n, residual = p.fit_contact(run, 2000)
    assert n == 7
    assert abs(z0 - 0.180) < 0.003
    assert 2.8e5 < stiffness < 3.3e5
    assert residual < 0.005  # the max_fit_error default holds for a real tap
    assert p.fit_contact(run[:3], 2000) is None


def test_board_debug_output_decodes():
    # printed by the V23 after finalize_config on a cold boot; used to kill the worker
    fmt = "allocMax=%u usedMax=%u"
    parser = p.MessageParser({"output": {fmt: 92}})
    data = p.encode_vlq(92) + p.encode_vlq(3122) + p.encode_vlq(0)
    assert parser.decode(data) == [(fmt, {"arg0": 3122, "arg1": 0})]


def test_message_parser():
    parser = p.MessageParser(
        {
            "commands": {
                "get_uptime": 7,
                "stop_prtouch_pres oid=%c sta_swap=%c": 9,
            },
            "responses": {"ack_prtouch oid=%c err=%c expar0=%u expar1=%u": -24},
            "enumerations": {"pin": {"PA0": [0, 16], "PB0": [16, 16]}},
        }
    )
    assert parser.pin("PB10") == 26 and parser.pin("PA8") == 8
    assert parser.encode("stop_prtouch_pres", oid=0, sta_swap=1) == bytes(
        [9, 0, 1]
    )
    reply = p.encode_vlq(-24) + bytes([0, 0, 0, 84])
    assert parser.decode(reply) == [
        ("ack_prtouch", {"oid": 0, "err": 0, "expar0": 0, "expar1": 84})
    ]


def test_command_names_have_no_digits():
    # Digits split extended G-code names: V23_PROBE is parsed as V23.
    names = [
        n[len("cmd_") :]
        for n in vars(p.V23LoadCell)
        if n.startswith("cmd_") and not n.endswith("_help")
    ]
    assert names and not any(c.isdigit() for n in names for c in n)


def test_resolve_backoff_relieves_before_retare_on_compliant_bed():
    # Stiffness 60 counts/um: 10000-count trigger_force needs ~0.167mm past
    # contact, more than the old fixed 3*coarse_step=0.15mm backoff could
    # clear. resolve_backoff must keep stepping until force actually drops.
    stiffness = 60000.0  # counts/mm
    depth = [10000.0 / stiffness + 0.02]  # a bit past the trigger threshold
    step_calls = []

    def measure_force():
        return max(0.0, stiffness * depth[0])

    def step(dz):
        step_calls.append(dz)
        depth[0] = max(0.0, depth[0] - dz)

    force = p.resolve_backoff(
        measure_force, step, size=0.05, max_back=1.0, fit_min_force=2000
    )
    assert force <= 2000
    assert len(step_calls) > 3  # the old single 3x-step backoff undershot


def test_resolve_backoff_reports_unrelieved_force_within_bound():
    # A bed that never relieves (e.g. something still pressed) must not be
    # silently accepted as a tare baseline; the caller treats this as a
    # fault when the returned force is still over fit_min_force.
    def measure_force():
        return 999999

    def step(dz):
        pass

    force = p.resolve_backoff(
        measure_force, step, size=0.05, max_back=0.15, fit_min_force=2000
    )
    assert force > 2000


def test_add_samples_backdates_older_samples_within_a_late_batch(monkeypatch):
    # A batch read late (e.g. after a backlog) can span real acquisition
    # time the host never saw. The newest sample is anchored to receipt
    # time (the best available estimate), but older samples in the same
    # batch must not share that timestamp: that was the original bug
    # (a whole batch stamped "now" looks uniformly fresh regardless of
    # how much real time it actually spans).
    w = p.BoardWorker(None)
    w.acq_period = 0.001
    w.seen = [p.DISCARD_SAMPLES] * p.CHANNELS_MAX  # past the startup discard
    monkeypatch.setattr(p.time, "monotonic", lambda: 1000.0)
    packed = (
        bytes([2, 0])
        + (10).to_bytes(1, "little", signed=True)
        + (5).to_bytes(1, "little", signed=True)
    )
    w._add_samples({"ch": 0, "datas": packed})
    timestamps = [t for t, _ in w.samples[0]]
    assert timestamps == [1000.0 - 0.001, 1000.0]  # oldest first, newest = now


def test_add_samples_does_not_drift_from_wall_clock_across_batches(monkeypatch):
    # Each batch re-anchors to its own receipt time rather than counting
    # forward from when streaming started, so timestamps can't drift away
    # from wall-clock if the real rate differs slightly from acq_period
    # (the bug that made the very first post-fix hardware test fail: a
    # long-running synthetic clock based on an assumed rate fell behind
    # real time, so every sample looked stale forever).
    w = p.BoardWorker(None)
    w.acq_period = 0.001
    w.seen = [p.DISCARD_SAMPLES] * p.CHANNELS_MAX
    one_sample = bytes([1, 0]) + (1).to_bytes(1, "little", signed=True)
    for fake_now in (100.0, 500.0, 1000.0):
        monkeypatch.setattr(p.time, "monotonic", lambda t=fake_now: t)
        w._add_samples({"ch": 0, "datas": one_sample})
    timestamps = [t for t, _ in w.samples[0]]
    assert timestamps == [100.0, 500.0, 1000.0]


def test_channel_means_requires_minimum_coverage():
    w = p.BoardWorker(None)
    w.samples[0] = [(10.0, 1.0), (10.001, 1.0)]
    assert w.channel_means(since=9.0, channels=1, min_samples=5) is None
    w.samples[0] += [(10.002, 1.0), (10.003, 1.0), (10.004, 1.0)]
    assert w.channel_means(since=9.0, channels=1, min_samples=5) == [1.0]


def test_drain_discards_samples_read_right_after_a_reader_stall(monkeypatch):
    # Backdating within a batch (_add_samples) only accounts for the batch's
    # own span, not time the whole batch spent queued before the host got
    # around to reading it. A gap between drain cycles means exactly that
    # may have happened, so those samples must be dropped, not merely
    # backdated by their own (now understated) span.
    class _StubParser:
        def decode(self, payload):
            data = bytes([1, 0]) + (1).to_bytes(1, "little", signed=True)
            return [("resault_prtouch_pres", {"ch": 0, "datas": data})]

    class _StubLink:
        parser = _StubParser()

        def read_payloads(self, timeout):
            yield b"\x00"

    w = p.BoardWorker(None)
    w.acq_period = 0.001
    w.seen = [p.DISCARD_SAMPLES] * p.CHANNELS_MAX  # past the startup discard
    w.link = _StubLink()
    t = [1000.0]
    monkeypatch.setattr(p.time, "monotonic", lambda: t[0])

    w._drain()  # first call: no prior drain to compare against
    assert len(w.samples[0]) == 1

    t[0] += p.DRAIN_STALL_THRESHOLD + 0.1  # simulate a scheduling stall
    w._drain()
    assert len(w.samples[0]) == 1  # the post-stall batch was dropped

    t[0] += 0.01  # cadence back to normal
    w._drain()
    assert len(w.samples[0]) == 2  # normal reads resume being recorded


def test_drain_discards_samples_from_a_stall_inside_a_single_call(monkeypatch):
    # A stall can happen *during* read_payloads() itself (e.g. the OS
    # suspends serial.read() mid-call), not just in the gap before this call
    # started. Checking only the pre-call gap misses this: the call still
    # starts on time, decodes a small-looking batch, and by the time the
    # *next* call detects the elapsed gap, this one's samples are already
    # recorded.
    class _StubParser:
        def decode(self, payload):
            data = bytes([1, 0]) + (1).to_bytes(1, "little", signed=True)
            return [("resault_prtouch_pres", {"ch": 0, "datas": data})]

    t = [1000.0]

    class _StubLink:
        parser = _StubParser()

        def read_payloads(self, timeout):
            yield b"\x00"  # read before the stall
            t[0] += p.DRAIN_STALL_THRESHOLD + 0.1  # the stall happens here
            yield b"\x00"  # decoded only once the read resumes

    w = p.BoardWorker(None)
    w.acq_period = 0.001
    w.seen = [p.DISCARD_SAMPLES] * p.CHANNELS_MAX
    w.link = _StubLink()
    monkeypatch.setattr(p.time, "monotonic", lambda: t[0])

    w._drain()
    assert len(w.samples[0]) == 0  # both payloads from this call are dropped


class _FakeSerial:
    def __init__(self):
        self.written = []

    def write(self, data):
        self.written.append(data)

    def close(self):
        pass


class _FakeParser:
    dictionary = {"config": {"CLOCK_FREQ": 144000000}}

    def encode(self, name, **kwargs):
        return b"ENC:" + name.encode()

    def pin(self, name):
        return 0


class _FakeLink:
    def __init__(self, fail_query=False):
        self.parser = _FakeParser()
        self.seq = 0
        self.ser = _FakeSerial()
        self.fail_query = fail_query

    def query(self, *args, **kwargs):
        if self.fail_query:
            raise p.serial.SerialException("no reply")

    def close(self):
        pass


def test_stop_and_close_still_attempts_stop_when_query_fails():
    w = p.BoardWorker(None)
    fake = _FakeLink(fail_query=True)
    w.link = fake
    w.streaming = True
    w.board_streaming = True
    with pytest.raises(p.serial.SerialException):
        w.stop_and_close()
    assert fake.ser.written  # close()'s fallback stop was still sent
    assert w.link is None
    assert w.board_streaming is False


class _ExplodingLink:
    def __init__(self):
        self.parser = _FakeParser()
        self.seq = 0
        self.ser = _FakeSerial()

    def read_payloads(self, timeout):
        raise p.serial.SerialException("link lost")
        yield  # pragma: no cover - makes this a generator

    def close(self):
        pass


def test_drain_link_loss_keeps_board_streaming_flag_for_close():
    w = p.BoardWorker(None)
    fake = _ExplodingLink()
    w.link = fake
    w.streaming = True
    w.board_streaming = True
    w._drain()
    assert w.streaming is False  # the drain loop itself stops
    assert w.board_streaming is True  # but the MCU might still be emitting
    w.close()
    assert fake.ser.written  # close() still attempted the stop


class _FakeOpenStartLink:
    """identify/get_config succeed, but start_prtouch_pres's ack is lost."""

    def __init__(self, *args, **kwargs):
        self.parser = _FakeParser()
        self.seq = 0
        self.ser = _FakeSerial()

    def identify(self, tries=4):
        return self.parser

    def query(self, name, want, **kwargs):
        if name == "get_config":
            return {"is_config": True, "crc": p.CONFIG_CRC}
        raise p.serial.SerialException("ack lost for %s" % name)

    def send(self, payload):
        pass

    def close(self):
        pass


class _FakeV23Config:
    serial_port = "/dev/fake"
    baud = 230400
    cfg_regs = 60
    swap_pin = "PA8"
    channel_pins = [("PB10", "PB2")]


def test_open_and_start_marks_board_streaming_before_the_ack(monkeypatch):
    # If the MCU receives start_prtouch_pres but its ack is lost (or the
    # request is cancelled waiting for one), the board may already be
    # streaming even though open_and_start() ends up raising. board_streaming
    # must reflect that *before* the query, not after it succeeds, or the
    # except-clause's close() below has nothing telling it to send a stop.
    created = {}

    def fake_link(*args, **kwargs):
        link = _FakeOpenStartLink()
        created["link"] = link
        return link

    monkeypatch.setattr(p, "BoardLink", fake_link)
    w = p.BoardWorker(_FakeV23Config())
    with pytest.raises(p.serial.SerialException):
        w.open_and_start()
    assert created["link"].ser.written  # close()'s fallback stop was sent
    assert w.board_streaming is False  # cleared after close() attempted it


def test_native_crc_matches_independent_reflected_polynomial():
    def reference(data):
        crc = 0xFFFF
        for value in data:
            crc ^= value
            for _ in range(8):
                crc = (crc >> 1) ^ (0x8408 if crc & 1 else 0)
        return crc

    assert p.crc16(b"123456789") == 0x6F91
    assert p.crc16(b"") == 0xFFFF
    rng = random.Random(0)
    cases = [bytes([value]) for value in range(256)]
    cases += [
        bytes(rng.randrange(256) for _ in range(length))
        for length in range(p.MSG_MAX + 1)
        for _ in range(4)
    ]
    for data in cases:
        assert p.crc16(data) == reference(data)


def test_crc_still_rejects_corrupt_frame_and_recovers():
    valid = p.build_block(3, b"abc")
    corrupt = bytearray(valid)
    corrupt[3] ^= 0x40
    # Supply enough subsequent traffic to resolve any plausible partial
    # frame length encountered while resynchronizing past the corrupt frame.
    payloads, sequences, remaining = p.split_blocks(bytes(corrupt) + valid * 10)
    assert payloads == [b"abc"] * 10
    assert sequences == [3] * 10
    assert remaining == b""


def measurement_probe(monkeypatch, feed=None):
    clock = SimpleNamespace(now=10.0, pauses=[])
    monkeypatch.setattr(p.time, "monotonic", lambda: clock.now)
    probe = object.__new__(p.V23LoadCell)
    probe.settle_time = probe.sample_time = 0.15
    probe.sample_timeout = 0.5
    probe.channel_pins = [None] * 4
    probe.gcode = SimpleNamespace(error=RuntimeError)
    probe.worker = p.BoardWorker(None)
    probe.worker.streaming = True
    probe.worker.acq_period = 0.00078125

    def pause(duration):
        assert duration >= 0
        clock.pauses.append(duration)
        clock.now += duration
        if feed is not None:
            feed(probe.worker, clock.now)

    probe._pause = pause
    return probe, clock


def fresh_samples(at, value, count=5):
    return [(at - index * 0.001, value) for index in range(count)]


def test_measure_waits_for_delayed_channel_after_settling(monkeypatch):
    def feed(worker, now):
        for ch in range(4):
            at = now if ch or now >= 10.35 else 10.05
            worker.samples[ch] = fresh_samples(at, ch + 1)

    probe, clock = measurement_probe(monkeypatch, feed)
    assert probe._measure() == [1, 2, 3, 4]
    assert clock.now == pytest.approx(10.36)
    assert max(clock.pauses[2:]) <= p.MEASUREMENT_RETRY_INTERVAL


def test_measure_never_reuses_pre_settle_data(monkeypatch):
    probe, clock = measurement_probe(monkeypatch)
    probe.worker.samples = [fresh_samples(10.1, 42) for _ in range(4)]
    with pytest.raises(RuntimeError, match="ch0=0 samples"):
        probe._measure()
    assert clock.now == pytest.approx(10.65)


def test_measure_rolling_window_excludes_channels_that_went_stale(monkeypatch):
    def feed(worker, now):
        worker.samples[0] = fresh_samples(now, 1) if now >= 10.55 else []
        for ch in (1, 2, 3):
            worker.samples[ch] = fresh_samples(min(now, 10.3), ch + 1)

    probe, clock = measurement_probe(monkeypatch, feed)
    with pytest.raises(RuntimeError, match="ch1=0 samples") as error:
        probe._measure()
    assert "ch0=5 samples" in str(error.value)
    assert "window 0.150s" in str(error.value)
    assert clock.now == pytest.approx(10.65)


def test_measure_timeout_reports_all_missing_channels(monkeypatch):
    probe, clock = measurement_probe(monkeypatch)
    with pytest.raises(RuntimeError, match="within 0.500s") as error:
        probe._measure()
    for ch in range(4):
        assert "ch%d=0 samples (age none)" % ch in str(error.value)
    assert clock.now == pytest.approx(10.65)


def test_measure_does_not_accept_data_after_slow_query_exceeds_deadline(
    monkeypatch,
):
    def feed(worker, now):
        worker.samples = [fresh_samples(now, ch + 1) for ch in range(4)]

    probe, clock = measurement_probe(monkeypatch, feed)
    query = probe.worker.channel_measurement

    def delayed_query(*args, **kwargs):
        result = query(*args, **kwargs)
        clock.now += 0.4
        return result

    probe.worker.channel_measurement = delayed_query
    with pytest.raises(RuntimeError, match="within 0.500s"):
        probe._measure()
    assert clock.now == pytest.approx(10.7)


def test_measure_rejects_reactor_oversleep_before_first_query(monkeypatch):
    probe, clock = measurement_probe(monkeypatch)
    pause = probe._pause

    def delayed_pause(duration):
        pause(duration)
        if len(clock.pauses) == 2:
            clock.now += 1.0

    probe._pause = delayed_pause
    with pytest.raises(RuntimeError, match="no snapshot completed"):
        probe._measure()


@pytest.mark.parametrize("shutdown", [False, True])
def test_measure_aborts_when_stream_stops_while_waiting(monkeypatch, shutdown):
    def feed(worker, now):
        if now > 10.31:
            if shutdown:
                worker.stopping.set()
            else:
                worker.streaming = False

    probe, clock = measurement_probe(monkeypatch, feed)
    with pytest.raises(RuntimeError, match="board stream stopped"):
        probe._measure()
    assert clock.now < 10.65


def test_channel_measurement_handles_overlapping_batch_timestamps(monkeypatch):
    monkeypatch.setattr(p.time, "monotonic", lambda: 10.3)
    worker = p.BoardWorker(None)
    worker.acq_period = 0.001
    worker.samples[0] = [
        (10.18, 1),
        (10.25, 2),
        (10.14, 1000),
        (10.26, 3),
        (10.27, 4),
        (10.28, 5),
        (10.5, 1000),
    ]
    means, coverage = worker.channel_measurement(10.16, 1, 5, window=0.15)
    assert means == [3]
    assert coverage[0][0] == 5


def test_channel_measurement_overlapping_packets_match_full_history_scan(
    monkeypatch,
):
    worker = p.BoardWorker(None)
    worker.acq_period = 0.001
    worker.seen = [p.DISCARD_SAMPLES] * p.CHANNELS_MAX
    now = [10.0]
    monkeypatch.setattr(p.time, "monotonic", lambda: now[0])
    # Build valid compressed batches whose timestamp ranges overlap. A
    # reverse scan must not stop at the first old sample within a later batch.
    data = bytes([20]) + bytes(5) + bytes([1]) * 20
    for i in range(500):
        now[0] = 10.0 + i * 0.01
        for ch in range(4):
            worker._add_samples({"ch": ch, "datas": data})
    for age in (0.01, 0.015, 0.05, 0.15, 0.4):
        cutoff = now[0] - age
        expected = worker.channel_means(cutoff, 4, 5)
        actual, coverage = worker.channel_measurement(10.0, 4, 5, window=age)
        assert actual == expected
        assert all(count >= 5 for count, _ in coverage)

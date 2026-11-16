import pytest

from klippy.extras import prtouch_v3 as p


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
    # klippy's gcode parser splits "PRTOUCH_V3_PROBE" into "PRTOUCH_V3" + junk
    names = [
        n[len("cmd_") :]
        for n in vars(p.PrTouchV3)
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


class _FakeSerial:
    def __init__(self):
        self.written = []

    def write(self, data):
        self.written.append(data)

    def close(self):
        pass


class _FakeParser:
    def encode(self, name, **kwargs):
        return b"ENC:" + name.encode()


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

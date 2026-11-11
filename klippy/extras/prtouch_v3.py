# Z homing with Creality prtouch_v3 load cell boards (K1-MAX-L_V23)
#
# Copyright (C) 2026 gilrim
#
# This file may be distributed under the terms of the GNU GPLv3 license.
#
# The K1-MAX-L_V23 leveling board (CH32V203 + 4x CS1237) runs Creality's own
# Klipper build ("CTF_V1.0.0.16.20250320") with the prtouch_v3 command set. It
# can't run Kalico firmware, and a Klipper `reset` parks it in an IAP
# bootloader for ~60s, so it is not configured as an [mcu]. This module owns
# its serial port instead, from a worker thread so the reactor never blocks.
#
# Probing: tare the summed force of the four cells, step Z down (coarse, back
# off, fine) reading the settled force after each step, and fit a line through
# the fine steps past contact. The bed is compliant, so force rises linearly
# with depth; the zero-force intercept of that line is the contact height.
# Measured on a K1 Max: ~300 counts/um stiffness, 1.1um spread over 3 probes.
#
# Protocol reconstructed from Creality's prtouch_v3_wrapper (X2000E firmware
# V1.1.0.27) and the board's data dictionary.
import json
import logging
import math
import queue
import re
import threading
import time
import zlib
from collections import deque

import serial

MSG_MIN, MSG_MAX, MSG_SYNC = 5, 64, 0x7E
IDENTIFY_CHUNK = 40
# prtouch_v3 defaults for CS1237 boards (pres_cfg_regs 0x3C = 1280Hz/PGA128):
# acq_tkms 0.78125ms when cfg_regs & 0x30 == 0x30, ded_tkms 128
DEFAULT_CFG_REGS = 60
ACQ_MS_FAST, ACQ_MS_SLOW = 0.78125, 1.5625
DEAD_MS = 128
DISCARD_SAMPLES = 20  # first samples after start can read doubled
CONFIG_CRC = 0x7623  # finalize_config crc, to recognise our own config
CHANNELS_MAX = 4


######################################################################
# Klipper wire protocol (minimal, host side)
######################################################################


def crc16(data):
    crc = 0xFFFF
    for b in data:
        b ^= crc & 0xFF
        b ^= (b & 0x0F) << 4
        crc = ((b << 8) | (crc >> 8)) ^ (b >> 4) ^ (b << 3)
        crc &= 0xFFFF
    return crc


def encode_vlq(v):
    out = []
    if not -(1 << 26) <= v < (3 << 26):
        out.append((v >> 28) & 0x7F | 0x80)
    if not -(1 << 19) <= v < (3 << 19):
        out.append((v >> 21) & 0x7F | 0x80)
    if not -(1 << 12) <= v < (3 << 12):
        out.append((v >> 14) & 0x7F | 0x80)
    if not -(1 << 5) <= v < (3 << 5):
        out.append((v >> 7) & 0x7F | 0x80)
    out.append(v & 0x7F)
    return bytes(out)


def decode_vlq(buf, pos):
    c = buf[pos]
    pos += 1
    v = c & 0x7F
    if c & 0x60 == 0x60:
        v |= -0x20
    while c & 0x80:
        c = buf[pos]
        pos += 1
        v = (v << 7) | (c & 0x7F)
    return v, pos


def build_block(seq, payload):
    blk = bytes([len(payload) + MSG_MIN, 0x10 | (seq & 0x0F)]) + payload
    c = crc16(blk)
    return blk + bytes([c >> 8, c & 0xFF, MSG_SYNC])


def split_blocks(buf):
    """Return (payloads, sequence nibbles, leftover bytes)."""
    payloads, seqs = [], []
    while buf:
        n = buf[0]
        if not MSG_MIN <= n <= MSG_MAX:
            buf = buf[1:]
            continue
        if len(buf) < n:
            break
        blk = buf[:n]
        if blk[-1] == MSG_SYNC and crc16(blk[:-3]) == (blk[-3] << 8 | blk[-2]):
            payloads.append(blk[2:-3])
            seqs.append(blk[1] & 0x0F)
            buf = buf[n:]
        else:
            buf = buf[1:]
    return payloads, seqs, buf


WIDTHS = {"%c": (8, False), "%hu": (16, False), "%hi": (16, True)}
WIDTHS.update({"%u": (32, False), "%i": (32, True)})


def fix_width(v, fmt):
    bits, signed = WIDTHS.get(fmt, (32, True))
    v &= (1 << bits) - 1
    return v - (1 << bits) if signed and v >> (bits - 1) else v


def parse_format(fmt):
    parts = fmt.split()
    return parts[0], [p.split("=", 1) for p in parts[1:]]


def parse_output(fmt):
    # printf-style debug text ("Timer too close waketime = %u,..."): the
    # params are just the % conversions, in order
    convs = re.findall(r"%[-0-9.*]*[a-z]+", fmt)
    return fmt, [("arg%d" % i, c) for i, c in enumerate(convs)]


class MessageParser:
    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.commands = {}
        self.responses = {}
        for fmt, msgid in dictionary.get("commands", {}).items():
            name, params = parse_format(fmt)
            self.commands[name] = (msgid, params)
        for fmt, msgid in dictionary.get("responses", {}).items():
            self.responses[msgid] = parse_format(fmt)
        for fmt, msgid in dictionary.get("output", {}).items():
            self.responses[msgid] = parse_output(fmt)

    def pin(self, name):
        ranges = self.dictionary.get("enumerations", {}).get("pin", {})
        for base, (start, count) in ranges.items():
            port, first = base[:-1], int(base[-1])
            num = name[len(port) :]
            if name.startswith(port) and num.isdigit():
                idx = int(num) - first
                if 0 <= idx < count:
                    return start + idx
        raise ValueError(f"unknown pin {name}")

    def encode(self, name, **args):
        if name not in self.commands:
            raise ValueError(f"board firmware has no '{name}' command")
        msgid, params = self.commands[name]
        out = encode_vlq(msgid)
        for pname, fmt in params:
            v = args[pname]
            if "s" in fmt:
                out += encode_vlq(len(v)) + bytes(v)
            else:
                out += encode_vlq(int(v))
        return out

    def decode(self, payload):
        msgs, pos = [], 0
        while pos < len(payload):
            msgid, pos = decode_vlq(payload, pos)
            if msgid not in self.responses:
                break
            name, params = self.responses[msgid]
            vals = {}
            for pname, fmt in params:
                if "s" in fmt:
                    ln, pos = decode_vlq(payload, pos)
                    vals[pname] = bytes(payload[pos : pos + ln])
                    pos += ln
                else:
                    v, pos = decode_vlq(payload, pos)
                    vals[pname] = fix_width(v, fmt)
            msgs.append((name, vals))
        return msgs


def unzip_data(data, signed):
    """prtouch sample packing: data[0] = count, 2-bit length codes (stored
    from the end of the header backwards), little endian delta coded values.
    """
    n = data[0]
    ptr = int(math.ceil(n / 4.0)) + 1
    idx, out = ptr, []
    for i in range(n):
        code = data[ptr - ((i >> 2) + 1)] >> ((i & 3) << 1)
        v_len = 1 + (code & 3)
        v = int.from_bytes(data[idx : idx + v_len], "little", signed=True)
        out.append(v + (out[-1] if out else 0))
        idx += v_len
    return out if signed else [v & 0xFFFFFFFF for v in out]


def fit_contact(points, fit_min):
    """Fit force = k * (z0 - z) through the points above fit_min and return
    (z0, k per mm, n, maximum residual in mm), or None for invalid data."""
    if any(not math.isfinite(v) for point in points for v in point):
        return None
    pts = [(z, f) for z, f in points if f > fit_min]
    n = len(pts)
    if n < 3:
        return None
    mz = sum(z for z, _ in pts) / n
    mf = sum(f for _, f in pts) / n
    sxx = sum((z - mz) ** 2 for z, _ in pts)
    if not sxx:
        return None
    slope = sum((z - mz) * (f - mf) for z, f in pts) / sxx
    if slope >= 0:
        return None
    error = max(abs(f - (mf + slope * (z - mz))) for z, f in pts)
    return mz - mf / slope, -slope, n, error / -slope


class BoardLink:
    """Blocking serial session with the board. Only used from the worker."""

    def __init__(self, port, baud, check_active):
        self.check_active = check_active
        self.ser = serial.Serial(
            port, baud, timeout=0.05, write_timeout=0.1, exclusive=True
        )
        self.seq = 0
        self.mcu_seq = None
        self.buf = b""
        self.pending = deque()
        self.parser = None
        try:
            self.check_active()
            self.ser.write(bytes([MSG_SYNC]) * 3)
            time.sleep(0.05)
            self.ser.reset_input_buffer()
        except Exception:
            self.ser.close()
            raise

    def close(self):
        self.ser.close()

    def send(self, payload):
        self.check_active()
        self.ser.write(build_block(self.seq, payload))
        self.seq += 1

    def read_payloads(self, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.check_active()
            while self.pending:
                yield self.pending.popleft()
            self.buf += self.ser.read(max(256, self.ser.in_waiting))
            payloads, seqs, self.buf = split_blocks(self.buf)
            if seqs:
                self.mcu_seq = seqs[-1]
            # Keep unconsumed replies if a query returns from this generator.
            self.pending.extend(p for p in payloads if p)
            while self.pending:
                yield self.pending.popleft()

    def resync(self):
        # The board keeps its sequence counter across sessions and nak's a
        # stale one; adopt the sequence it last asked for.
        if self.mcu_seq is not None:
            self.seq = self.mcu_seq

    def identify(self, tries=4):
        for _ in range(tries):
            data, offset = b"", 0
            while True:
                self.send(
                    encode_vlq(1)
                    + encode_vlq(offset)
                    + encode_vlq(IDENTIFY_CHUNK)
                )
                chunk = None
                for p in self.read_payloads(0.5):
                    try:
                        msgid, pos = decode_vlq(p, 0)
                        if msgid != 0:
                            continue
                        roffset, pos = decode_vlq(p, pos)
                        ln, pos = decode_vlq(p, pos)
                    except IndexError:
                        continue
                    if roffset == offset:
                        chunk = p[pos : pos + ln]
                        break
                if chunk is None:
                    break
                if not chunk:
                    self.parser = MessageParser(
                        json.loads(zlib.decompress(data))
                    )
                    return self.parser
                data += chunk
                offset += len(chunk)
            self.ser.write(bytes([MSG_SYNC]) * 3)
            time.sleep(0.1)
            self.resync()
        raise serial.SerialException("no identify response from board")

    def query(self, name, want, tries=3, timeout=1.0, **args):
        for _ in range(tries):
            self.send(self.parser.encode(name, **args))
            for p in self.read_payloads(timeout):
                try:
                    msgs = self.parser.decode(p)
                except IndexError:
                    continue
                for mname, vals in msgs:
                    if mname == want:
                        return vals
            self.resync()
        raise serial.SerialException(f"no '{want}' reply to '{name}'")


######################################################################
# Worker thread: owns the serial port
######################################################################


class _Request:
    def __init__(self, func, timeout):
        self.func = func
        self.deadline = time.monotonic() + timeout
        self.result = self.error = None
        self.done = threading.Event()
        self.cancelled = threading.Event()


class BoardWorker(threading.Thread):
    def __init__(self, prtouch):
        super().__init__(daemon=True)
        self.prtouch = prtouch
        self.requests = queue.Queue()
        self.stopping = threading.Event()
        self.active = None
        self.link = None
        self.streaming = False
        self.lock = threading.Lock()
        self.samples = [[] for _ in range(CHANNELS_MAX)]
        self.seen = [0] * CHANNELS_MAX

    def check_active(self):
        req = self.active
        if self.stopping.is_set() or (
            req is not None
            and (req.cancelled.is_set() or time.monotonic() >= req.deadline)
        ):
            raise serial.SerialException("board request cancelled or timed out")

    def shutdown(self):
        self.stopping.set()

    def run(self):
        try:
            while not self.stopping.is_set():
                # Never block while streaming: the board sends ~15KB/s and
                # any gap in reading backs up the tty, delaying (and then
                # losing) samples, so readings lag behind the motion.
                try:
                    req = self.requests.get(
                        block=not self.streaming, timeout=0.05
                    )
                except queue.Empty:
                    req = None
                if req is not None:
                    self.active = req
                    try:
                        self.check_active()
                        req.result = req.func()
                    except Exception as e:
                        req.error = e
                        # A failed request must not leave a reusable session.
                        self.close()
                    finally:
                        self.active = None
                        req.done.set()
                if self.link is not None and self.streaming:
                    self._drain()
        except Exception:
            # Thread exceptions don't reach the klippy log otherwise
            logging.exception("prtouch_v3: board worker died")
            raise
        finally:
            self.close()
            while True:
                try:
                    req = self.requests.get_nowait()
                except queue.Empty:
                    break
                req.error = serial.SerialException("board worker disconnected")
                req.done.set()

    def _drain(self):
        try:
            for p in self.link.read_payloads(0.05):
                try:
                    msgs = self.link.parser.decode(p)
                except IndexError:
                    continue
                for name, vals in msgs:
                    if " " in name:  # board debug output
                        logging.info("prtouch_v3: board says %s %s", name, vals)
                    if name != "resault_prtouch_pres":
                        continue
                    # One bad packet must not end the worker thread
                    try:
                        self._add_samples(vals)
                    except (IndexError, KeyError, ValueError):
                        logging.exception("prtouch_v3: bad sample packet")
        except serial.SerialException:
            logging.exception("prtouch_v3: board link lost while streaming")
            self.streaming = False

    def _add_samples(self, vals):
        ch = vals["ch"]
        if ch >= CHANNELS_MAX:
            return
        datas = unzip_data(vals["datas"], True)
        skip = max(0, DISCARD_SAMPLES - self.seen[ch])
        self.seen[ch] += len(datas)
        now = time.monotonic()
        with self.lock:
            self.samples[ch].extend((now, d) for d in datas[skip:])
            del self.samples[ch][:-20000]

    def channel_means(self, since, channels):
        with self.lock:
            means = []
            for ch in range(channels):
                w = [d for t, d in self.samples[ch] if t >= since]
                if not w:
                    return None
                means.append(sum(w) / len(w))
            return means

    # Requests (run on the worker thread)
    def open_and_start(self):
        p = self.prtouch
        self.link = link = BoardLink(p.serial_port, p.baud, self.check_active)
        try:
            parser = link.identify()
            cfg = link.query("get_config", "config")
            if cfg["is_config"] and cfg["crc"] != CONFIG_CRC:
                raise serial.SerialException(
                    "board configured by something else (crc=%d); power"
                    " cycle it" % (cfg["crc"],)
                )
            if not cfg["is_config"]:
                swap = parser.pin(p.swap_pin)
                link.send(parser.encode("allocate_oids", count=1))
                for idx, (clk, sdo) in enumerate(p.channel_pins):
                    link.send(
                        parser.encode(
                            "config_prtouch_pres",
                            oid=0,
                            idx=idx,
                            swp_pin=swap,
                            clk_pin=parser.pin(clk),
                            sdo_pin=parser.pin(sdo),
                        )
                    )
                link.send(parser.encode("finalize_config", crc=CONFIG_CRC))
                time.sleep(0.2)
            freq = parser.dictionary["config"]["CLOCK_FREQ"]
            acq_ms = ACQ_MS_FAST if p.cfg_regs & 0x30 == 0x30 else ACQ_MS_SLOW
            with self.lock:
                self.samples = [[] for _ in range(CHANNELS_MAX)]
            self.seen = [0] * CHANNELS_MAX
            link.query(
                "start_prtouch_pres",
                "ack_prtouch",
                oid=0,
                cfg_regs=p.cfg_regs,
                acq_tick=round(freq * acq_ms / 1000.0),
                ned_tftr=0,
                ned_hftr=0,
                ned_lftr=0,
                min_hold=128,
                max_hold=128,
                add_hold=0,
                lmt_dead=DEAD_MS,
            )
            self.streaming = True
            return parser.dictionary.get("version")
        except Exception:
            self.close()
            raise

    def stop_and_close(self):
        if self.link is None:
            return
        self.streaming = False
        try:
            self.link.query(
                "stop_prtouch_pres", "ack_prtouch", oid=0, sta_swap=1
            )
        finally:
            self.close()

    def close(self):
        link = self.link
        if link is not None and self.streaming and link.parser is not None:
            # Never leave the board streaming: identify can't find its reply
            # among the sample frames, so the next session couldn't connect.
            # Fire and forget, this may run while the worker is exiting.
            try:
                link.ser.write(
                    build_block(
                        link.seq,
                        link.parser.encode(
                            "stop_prtouch_pres", oid=0, sta_swap=1
                        ),
                    )
                )
            except Exception:
                logging.exception("prtouch_v3: stopping stream on close")
        self.streaming = False
        if self.link is not None:
            try:
                self.link.close()
            except Exception:
                logging.exception("prtouch_v3: closing serial port failed")
            finally:
                self.link = None


######################################################################
# The [prtouch_v3] module
######################################################################


class PrTouchV3:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.reactor = self.printer.get_reactor()
        self.gcode = self.printer.lookup_object("gcode")
        self.serial_port = config.get("serial")
        self.baud = config.getint("baud", 230400, minval=2400)
        self.swap_pin = config.get("swap_pin")
        pairs = [p.strip() for p in config.get("channel_pins").split(",")]
        try:
            self.channel_pins = [tuple(p.split(":")) for p in pairs if p]
        except ValueError:
            raise config.error("prtouch_v3: channel_pins are clk:sdo pairs")
        if not 1 <= len(self.channel_pins) <= CHANNELS_MAX or any(
            len(p) != 2 for p in self.channel_pins
        ):
            raise config.error(
                "prtouch_v3: channel_pins needs 1 to 4 clk:sdo pin pairs"
            )
        self.cfg_regs = config.getint("cfg_regs", DEFAULT_CFG_REGS, minval=0)
        self.z_offset = config.getfloat("z_offset", 0.0)
        self.coarse_step = config.getfloat("coarse_step", 0.05, above=0.0)
        self.fine_step = config.getfloat("fine_step", 0.01, above=0.0)
        self.step_speed = config.getfloat("step_speed", 2.0, above=0.0)
        self.settle_time = config.getfloat("settle_time", 0.15, minval=0.0)
        self.sample_time = config.getfloat("sample_time", 0.15, above=0.0)
        self.trigger_force = config.getint("trigger_force", 10000, minval=1)
        self.fit_min_force = config.getint(
            "fit_min_force", 2000, minval=1, maxval=self.trigger_force - 1
        )
        self.force_safety_limit = config.getint(
            "force_safety_limit", 300000, minval=self.trigger_force + 1
        )
        self.max_travel = config.getfloat("max_travel", 3.0, above=0.0)
        # Stiffness is configured in counts/um, residuals in mm. These broad
        # bounds surround the measured ~300 counts/um; tune from clean taps.
        self.min_stiffness = config.getfloat("min_stiffness", 50.0, above=0.0)
        self.max_stiffness = config.getfloat(
            "max_stiffness", 1500.0, above=self.min_stiffness
        )
        self.max_fit_error = config.getfloat("max_fit_error", 0.005, above=0.0)
        self.max_extrapolation = config.getfloat(
            "max_extrapolation", 0.1, above=0.0
        )
        self.last_z_result = None
        self.last_home_z_result = None
        self.last_stiffness = None
        self.last_accuracy = None
        self.worker = BoardWorker(self)
        self.printer.register_event_handler("klippy:ready", self.worker.start)
        self.printer.register_event_handler("klippy:shutdown", self._shutdown)
        self.printer.register_event_handler(
            "klippy:disconnect", self._disconnect
        )
        self.printer.register_event_handler(
            "homing:home_rails_begin", self._invalidate_home
        )
        self.printer.register_event_handler(
            "stepper_enable:motor_off", self._invalidate_home
        )
        self.gcode.register_command(
            "PRTOUCH_PROBE",
            self.cmd_PRTOUCH_PROBE,
            desc=self.cmd_PRTOUCH_PROBE_help,
        )
        self.gcode.register_command(
            "PRTOUCH_DIAGNOSTIC",
            self.cmd_PRTOUCH_DIAGNOSTIC,
            desc=self.cmd_PRTOUCH_DIAGNOSTIC_help,
        )
        self.gcode.register_command(
            "PRTOUCH_PROBE_ACCURACY",
            self.cmd_PRTOUCH_PROBE_ACCURACY,
            desc="Report nozzle contact repeatability",
        )
        self.gcode.register_command(
            "PRTOUCH_Z_OFFSET_APPLY_PROBE",
            self.cmd_Z_OFFSET_APPLY_PROBE,
            desc="Adjust the load cell z_offset",
        )

    def _invalidate_home(self, *args):
        self.last_home_z_result = None

    def _shutdown(self):
        self._invalidate_home()
        self.worker.shutdown()

    def _disconnect(self):
        self._shutdown()
        if self.worker.is_alive():
            self.worker.join(1.0)
            if self.worker.is_alive():
                logging.error("prtouch_v3: serial worker failed to exit")

    def get_status(self, eventtime):
        return {
            "last_z_result": self.last_z_result,
            "last_home_z_result": self.last_home_z_result,
            "last_stiffness": self.last_stiffness,
            "last_accuracy": self.last_accuracy,
        }

    # Worker access: post a request, keep the reactor running until done
    def _call(self, func, timeout=10.0):
        if not self.worker.is_alive() or self.worker.stopping.is_set():
            raise self.gcode.error("prtouch_v3: board worker is disconnected")
        req = _Request(func, timeout)
        self.worker.requests.put(req)
        end = self.reactor.monotonic() + timeout
        try:
            while not req.done.is_set():
                now = self.reactor.pause(self.reactor.monotonic() + 0.02)
                if now > end:
                    raise self.gcode.error(
                        "prtouch_v3: board request timed out"
                    )
        finally:
            if not req.done.is_set():
                req.cancelled.set()
        if req.error is not None:
            raise self.gcode.error(f"prtouch_v3: {req.error}")
        return req.result

    def _pause(self, seconds):
        self.reactor.pause(self.reactor.monotonic() + seconds)

    def _measure(self):
        self._pause(self.settle_time)
        since = time.monotonic()
        self._pause(self.sample_time)
        means = self.worker.channel_means(since, len(self.channel_pins))
        if means is None:
            raise self.gcode.error("prtouch_v3: no samples from the board")
        return means

    @staticmethod
    def _force(means, tare):
        # Every cell reads more negative under load
        return -sum(m - t for m, t in zip(means, tare))

    def _start(self):
        version = self._call(self.worker.open_and_start)
        logging.info("prtouch_v3: streaming from %s", version)
        self._pause(0.3)

    def _stop(self):
        try:
            self._call(self.worker.stop_and_close)
            return True
        except Exception:
            logging.exception("prtouch_v3: stop failed")
            # Termination closes the port on the worker. Never wait for a
            # second request on an unresponsive worker or reuse its session.
            self.worker.shutdown()
            return False

    def run_probe(self, gcmd):
        toolhead = self.printer.lookup_object("toolhead")
        curtime = self.reactor.monotonic()
        if "z" not in toolhead.get_status(curtime)["homed_axes"]:
            raise gcmd.error("prtouch_v3: must home z before probing")

        def step(dz):
            pos = toolhead.get_position()
            toolhead.manual_move([None, None, pos[2] + dz], self.step_speed)
            toolhead.wait_moves()
            return toolhead.get_position()[2]

        start_z = toolhead.get_position()[2]
        error = None
        try:
            self._start()
            fine_points = self._approach(gcmd, step, start_z)
        except Exception as e:
            error = e
            raise
        finally:
            # Relieve contact before any serial cleanup. If shutdown has
            # disabled motion, preserve the original error and still close.
            try:
                step(start_z - toolhead.get_position()[2])
            except Exception:
                if error is None:
                    raise
                logging.exception("prtouch_v3: retract after failure failed")
            finally:
                stopped = self._stop()
        if not stopped:
            raise gcmd.error(
                "prtouch_v3: board cleanup failed; restart Klipper"
            )
        fit = fit_contact(fine_points, self.fit_min_force)
        if fit is None:
            raise gcmd.error(
                "prtouch_v3: too few points past contact to fit; lower"
                " fit_min_force or raise trigger_force"
            )
        z0, stiffness, count, residual = fit
        nearest = max(z for z, f in fine_points if f > self.fit_min_force)
        if (
            not self.min_stiffness <= stiffness / 1000.0 <= self.max_stiffness
            or residual > self.max_fit_error
            or not 0.0 <= z0 - nearest <= self.max_extrapolation
            or not start_z - self.max_travel <= z0 <= start_z
        ):
            raise gcmd.error(
                "prtouch_v3: unreliable contact fit (stiffness %.1f counts/um,"
                " residual %.4fmm, extrapolation %.4fmm)"
                % (stiffness / 1000.0, residual, z0 - nearest)
            )
        return z0, stiffness, count

    def _approach(self, gcmd, step, start_z):
        """Coarse steps to contact, back off, fine steps; return the fine
        (z, force) points."""
        tare = self._measure()
        fine_points = []
        z = start_z
        for fine in (False, True):
            size = self.fine_step if fine else self.coarse_step
            while True:
                if start_z - z + size > self.max_travel + 1.0e-9:
                    raise gcmd.error(
                        "prtouch_v3: no contact within %.2fmm"
                        % (self.max_travel,)
                    )
                z = step(-size)
                force = self._force(self._measure(), tare)
                logging.info("prtouch_v3: z=%.4f force=%d", z, force)
                if force > self.force_safety_limit:
                    raise gcmd.error(
                        "prtouch_v3: force %d over force_safety_limit %d"
                        % (force, self.force_safety_limit)
                    )
                if fine:
                    fine_points.append((z, force))
                if force > self.trigger_force:
                    break
            if not fine:
                # back off the bed and re-tare before the fine pass
                z = step(3.0 * size)
                tare = self._measure()
        return fine_points

    cmd_PRTOUCH_PROBE_help = "Probe nozzle contact with the bed load cells"

    def cmd_PRTOUCH_PROBE(self, gcmd):
        home = gcmd.get("HOME", "").lower()
        if home not in ("", "z"):
            raise gcmd.error("prtouch_v3: HOME must be Z")
        if home:
            self._invalidate_home()
        z0, stiffness, count = self.run_probe(gcmd)
        self.last_z_result = z0
        self.last_stiffness = stiffness / 1000.0
        gcmd.respond_info(
            "prtouch_v3: contact at z=%.4f (fit over %d steps, %.0f"
            " counts/um)" % (z0, count, self.last_stiffness)
        )
        if home:
            toolhead = self.printer.lookup_object("toolhead")
            toolhead.get_last_move_time()
            pos = toolhead.get_position()
            pos[2] = pos[2] - z0 + self.z_offset
            toolhead.set_position(pos, homing_axes="z")
            self.last_home_z_result = z0
            gcmd.respond_info("prtouch_v3: z homed to the contact point")

    def cmd_PRTOUCH_PROBE_ACCURACY(self, gcmd):
        count = gcmd.get_int("SAMPLES", 10, minval=2, maxval=100)
        self.last_accuracy = None
        # Each result is one physical tap, without changing homing status.
        zs = [self.run_probe(gcmd)[0] for _ in range(count)]
        mean = sum(zs) / count
        ordered = sorted(zs)
        median = (ordered[count // 2] + ordered[(count - 1) // 2]) / 2.0
        stddev = math.sqrt(sum((z - mean) ** 2 for z in zs) / count)
        self.last_accuracy = {
            "samples": count,
            "minimum": min(zs),
            "maximum": max(zs),
            "range": max(zs) - min(zs),
            "average": mean,
            "median": median,
            "standard_deviation": stddev,
        }
        gcmd.respond_info(
            "prtouch_v3: probe accuracy results: maximum %.6f, minimum %.6f,"
            " range %.6f, average %.6f, median %.6f, standard deviation %.6f"
            % (max(zs), min(zs), max(zs) - min(zs), mean, median, stddev)
        )

    def cmd_Z_OFFSET_APPLY_PROBE(self, gcmd):
        gcode_move = self.printer.lookup_object("gcode_move")
        offset = gcode_move.get_status()["homing_origin"].z
        if offset == 0:
            gcmd.respond_info("Nothing to do: Z Offset is 0")
            return
        value = self.z_offset - offset
        self.printer.lookup_object("configfile").set(
            "prtouch_v3", "z_offset", "%.3f" % value
        )
        gcmd.respond_info(
            "prtouch_v3: z_offset: %.3f\n"
            "The SAVE_CONFIG command will update the printer config file\n"
            "with the above and restart the printer." % value
        )

    cmd_PRTOUCH_DIAGNOSTIC_help = "Report load cell readings and noise"

    def cmd_PRTOUCH_DIAGNOSTIC(self, gcmd):
        duration = gcmd.get_float("TIME", 5.0, above=0.0, maxval=60.0)
        try:
            self._start()
            tare = self._measure()
            forces = []
            end = self.reactor.monotonic() + duration
            while self.reactor.monotonic() < end:
                forces.append(self._force(self._measure(), tare))
        finally:
            self._stop()
        msg = ["prtouch_v3: tare per channel:"]
        msg += ["  ch%d: %d" % (i, t) for i, t in enumerate(tare)]
        msg.append(
            "force over %.1fs: min %d max %d (counts, %d readings)"
            % (duration, min(forces), max(forces), len(forces))
        )
        gcmd.respond_info("\n".join(msg))


def load_config(config):
    return PrTouchV3(config)

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

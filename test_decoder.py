def decode_ver_sw(v):
    """
    Unpack PX4 firmware version from 32-bit integer.
    Bit layout: major | minor | patch | release_type, one byte each.
    """
    if v is None or v == "":
        return None
    try:
        v = int(v)
    except (TypeError, ValueError):
        return None
    return (v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF

# 15 raw ver_sw values from the original screening run (pilot/per_log.csv)
RAW = [
    17629951, 17499135, 17760320, 17760256, 17694720,
    17825856, 17891328, 17694975, 17760384, 17761535,
    17826047, 17630207, 17695743, 17367104, 17629952,
]

# Anchor values Prof. Pattanayak explicitly verified:
#   17694720 -> v1.14.0
#   17761280 -> v1.15.4
#   17564416 -> v1.12.3
# The others below are decoded by the SAME function, so they are not
# independent ground truth. They are here to document the current output
# and to catch regressions if the decoder changes.

ANCHORS = {
    17694720: (1, 14, 0, 0),
    17761280: (1, 15, 4, 0),
    17564416: (1, 12, 3, 0),
}

print("=== Anchors (verified by Prof. Pattanayak) ===")
ok = 0
for raw, expected in ANCHORS.items():
    got = decode_ver_sw(raw)
    status = "OK" if got == expected else "FAIL"
    if status == "OK":
        ok += 1
    print(f"  {raw:10d} -> {got}  expected {expected}  [{status}]")
print(f"  {ok}/{len(ANCHORS)} anchors pass")

print("\n=== Decoded output for the 15 screening-run values ===")
for raw in RAW:
    print(f"  {raw:10d} -> {decode_ver_sw(raw)}")
def decode_ver_sw(v):
    if v is None or v == "":
        return None
    v = int(v)
    return (v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF

# Test cases from Prof. Pattanayak + a few from the original run
tests = [
    (17694720, (1, 14, 0, 0)),   # he says this should be v1.14.0
    (17761280, (1, 15, 4, 0)),   # he says this should be v1.15.4
    (17564416, (1, 12, 3, 0)),   # he says this should be v1.12.3
    # Additional values from the original screening run (200-log funnel)
    (17564416, (1, 12, 3, 0)),
    (17564671, (1, 12, 3, 255)),
    (17630207, (1, 13, 2, 255)),
    (17694975, (1, 14, 1, 255)),
    (17695488, (1, 14, 1, 0)),
    (17695743, (1, 14, 1, 255)),
    (17760256, (1, 15, 1, 0)),
    (17760320, (1, 15, 1, 64)),
    (17760384, (1, 15, 1, 128)),
    (17761535, (1, 15, 2, 255)),
    (17825856, (1, 16, 2, 0)),
    (17826047, (1, 16, 2, 255)),
    (17891328, (1, 17, 1, 0)),
    (17891583, (1, 17, 1, 255)),
    (18350592, (1, 24, 24, 0)),
]

for raw, expected in tests:
    got = decode_ver_sw(raw)
    status = "OK" if got == expected else "FAIL"
    print(f"{raw:10d} -> {got}  expected {expected}  [{status}]")
def decode_ver_sw(v):
    if v is None or v == "":
        return None
    v = int(v)
    return (v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF

# Only the three values Prof. Pattanayak explicitly named
tests = [
    (17694720, (1, 14, 0, 0)),   # his example: v1.14.0
    (17761280, (1, 15, 4, 0)),   # his example: v1.15.4
    (17564416, (1, 12, 3, 0)),   # his example: v1.12.3
]

for raw, expected in tests:
    got = decode_ver_sw(raw)
    status = "OK" if got == expected else "FAIL"
    print(f"{raw:10d} -> {got}  expected {expected}  [{status}]")
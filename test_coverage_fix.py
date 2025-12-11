"""
Coverage fix test - sys.settrace ile
"""

import sys

# Test fonksiyonu
def add(a: int, b: int):
    if a < 0:
        return -1
    return a + b

print("=== sys.settrace ile coverage test ===\n")

# Test 1: Pozitif değerler
print("Test 1: add(5, 3)")
executed_lines_1 = set()

def trace1(frame, event, arg):
    if event == 'line' and frame.f_code.co_name == 'add':
        executed_lines_1.add(frame.f_lineno)
    return trace1

sys.settrace(trace1)
result1 = add(5, 3)
sys.settrace(None)

print(f"  Result: {result1}")
print(f"  Executed lines: {sorted(executed_lines_1)}")

# Test 2: Negatif değer (farklı branch)
print("\nTest 2: add(-5, 3)")
executed_lines_2 = set()

def trace2(frame, event, arg):
    if event == 'line' and frame.f_code.co_name == 'add':
        executed_lines_2.add(frame.f_lineno)
    return trace2

sys.settrace(trace2)
result2 = add(-5, 3)
sys.settrace(None)

print(f"  Result: {result2}")
print(f"  Executed lines: {sorted(executed_lines_2)}")

# Fark
new_in_test2 = executed_lines_2 - executed_lines_1
print(f"\n  New lines in test 2: {sorted(new_in_test2)}")

print("\n✓ Test tamamlandı!")
print("\nBeklenen:")
print("  Test 1 (pozitif): satır 9, 11 çalışmalı (if False, return a+b)")
print("  Test 2 (negatif): satır 9, 10 çalışmalı (if True, return -1)")

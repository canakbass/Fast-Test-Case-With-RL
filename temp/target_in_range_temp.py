"""Training code 10: Check if value is in range"""

def in_range(value: int, min_val: int, max_val: int):
    if min_val > max_val:
        return False
    return min_val <= value <= max_val

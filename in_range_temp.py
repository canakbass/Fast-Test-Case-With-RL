
def in_range(value: int, low: int, high: int):
    if low > high:
        low, high = high, low
    if value < low:
        return "below"
    elif value > high:
        return "above"
    else:
        return "in_range"


def calculator(a: int, b: int):
    if a > 0:
        if b > 0:
            return a + b
        else:
            return a - b
    else:
        return a * b

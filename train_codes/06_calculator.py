"""Training code 6: Calculator with operations"""

def calculator(a: float, b: float, op: int):
    if op == 1:
        return a + b
    elif op == 2:
        return a - b
    elif op == 3:
        return a * b
    elif op == 4:
        if b != 0:
            return a / b
        return 0
    else:
        return -1

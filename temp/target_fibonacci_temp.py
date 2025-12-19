"""Training code 8: Fibonacci sequence"""

def fibonacci(n: int):
    if n < 0:
        return -1
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

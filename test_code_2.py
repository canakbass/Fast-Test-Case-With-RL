"""Test file 2: Math operations"""


def factorial(n: int) -> int:
    """Calculate factorial."""
    if n < 0:
        raise ValueError("Negative number")
    if n == 0:
        return 1
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


def fibonacci(n: int) -> int:
    """Get nth Fibonacci number."""
    if n < 0:
        raise ValueError("Negative index")
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def is_prime(n: int) -> bool:
    """Check if number is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def gcd(a: int, b: int) -> int:
    """Greatest common divisor."""
    while b:
        a, b = b, a % b
    return abs(a)


def power(base: float, exp: int) -> float:
    """Calculate power."""
    if exp < 0:
        return 1 / (base ** abs(exp))
    return base ** exp

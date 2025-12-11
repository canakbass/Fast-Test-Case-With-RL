
def gcd(a: int, b: int):
    a = abs(a)
    b = abs(b)
    while b:
        a, b = b, a % b
    return a

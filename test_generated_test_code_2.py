"""
Auto-generated test cases using RL-based Test Generator
"""
import pytest
from target_code import factorial, fibonacci, is_prime, gcd, power

# ===== factorial: 6 tests (from 6 generated cases) =====

def test_factorial_case_1():
    result = factorial(0)
    assert result == 1

def test_factorial_case_2():
    result = factorial(1)
    assert result == 1

def test_factorial_case_3():
    result = factorial(2)
    assert result == 2

def test_factorial_exception_1():
    with pytest.raises(ValueError):
        factorial(-2)

def test_factorial_exception_2():
    with pytest.raises(ValueError):
        factorial(-1)

def test_factorial_exception_3():
    with pytest.raises(ValueError):
        factorial(-3)

# ===== fibonacci: 5 tests (from 5 generated cases) =====

def test_fibonacci_case_1():
    result = fibonacci(0)
    assert result == 0

def test_fibonacci_case_2():
    result = fibonacci(1)
    assert result == 1

def test_fibonacci_case_3():
    result = fibonacci(2)
    assert result == 1

def test_fibonacci_exception_1():
    with pytest.raises(ValueError):
        fibonacci(-2)

def test_fibonacci_exception_2():
    with pytest.raises(ValueError):
        fibonacci(-1)

# ===== is_prime: 5 tests (from 5 generated cases) =====

def test_is_prime_case_1():
    result = is_prime(-2)
    assert result == False

def test_is_prime_case_2():
    result = is_prime(-1)
    assert result == False

def test_is_prime_case_3():
    result = is_prime(0)
    assert result == False

def test_is_prime_case_4():
    result = is_prime(1)
    assert result == False

def test_is_prime_case_5():
    result = is_prime(2)
    assert result == True

# ===== gcd: 7 tests (from 7 generated cases) =====

def test_gcd_case_1():
    result = gcd(-2, -2)
    assert result == 2

def test_gcd_case_2():
    result = gcd(-1, -1)
    assert result == 1

def test_gcd_case_3():
    result = gcd(0, 0)
    assert result == 0

def test_gcd_case_4():
    result = gcd(1, 1)
    assert result == 1

def test_gcd_case_5():
    result = gcd(2, 2)
    assert result == 2

def test_gcd_case_6():
    result = gcd(-1, 0)
    assert result == 1

def test_gcd_case_7():
    result = gcd(0, -1)
    assert result == 1

# ===== power: 10 tests (from 12 generated cases) =====

def test_power_case_1():
    result = power(-2.0, -2)
    assert result == 0.25

def test_power_case_2():
    result = power(-1.0, -1)
    assert result == -1.0

def test_power_case_3():
    result = power(0.0, 0)
    assert result == 1.0

def test_power_case_4():
    result = power(1.0, 1)
    assert result == 1.0

def test_power_case_5():
    result = power(2.0, 2)
    assert result == 4.0

def test_power_case_6():
    result = power(-0.17071524262428284, 0)
    assert result == 1.0

def test_power_case_7():
    result = power(-2.058797597885132, 0)
    assert result == 1.0

def test_power_case_8():
    result = power(-0.27667415142059326, 0)
    assert result == 1.0

def test_power_case_9():
    result = power(-0.3784847855567932, 0)
    assert result == 1.0

def test_power_case_10():
    result = power(-0.7706791758537292, -1)
    assert result == -1.2975567931911975


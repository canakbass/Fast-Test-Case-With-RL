"""
Base RL Model Training Script v2
--------------------------------
Yeni mimari ile base model eğitimi.
"""

from rl_module_v2 import train_base_model, generate_test_cases

# Çeşitli fonksiyonlardan oluşan dataset
dataset = [
    # 1. Basit matematik
    ("""
def add(a: int, b: int):
    return a + b
""", "add"),

    # 2. Karşılaştırma ve dallanma
    ("""
def max_of_three(a: int, b: int, c: int):
    if a >= b and a >= c:
        return a
    elif b >= a and b >= c:
        return b
    else:
        return c
""", "max_of_three"),

    # 3. Döngü
    ("""
def factorial(n: int):
    if n < 0:
        return -1
    if n == 0:
        return 1
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
""", "factorial"),

    # 4. Koşullu döngü
    ("""
def is_prime(n: int):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True
""", "is_prime"),

    # 5. Çoklu dallanma
    ("""
def grade(score: int):
    if score < 0 or score > 100:
        return "Invalid"
    elif score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"
""", "grade"),

    # 6. Matematik operasyonları
    ("""
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
""", "calculator"),

    # 7. GCD
    ("""
def gcd(a: int, b: int):
    a = abs(a)
    b = abs(b)
    while b:
        a, b = b, a % b
    return a
""", "gcd"),

    # 8. Fibonacci
    ("""
def fibonacci(n: int):
    if n < 0:
        return -1
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
""", "fibonacci"),

    # 9. Sayı kontrolü
    ("""
def check_number(n: int):
    if n > 0:
        if n % 2 == 0:
            return "positive_even"
        else:
            return "positive_odd"
    elif n < 0:
        if n % 2 == 0:
            return "negative_even"
        else:
            return "negative_odd"
    else:
        return "zero"
""", "check_number"),

    # 10. Range checker
    ("""
def in_range(value: int, low: int, high: int):
    if low > high:
        low, high = high, low
    if value < low:
        return "below"
    elif value > high:
        return "above"
    else:
        return "in_range"
""", "in_range"),
]

if __name__ == "__main__":
    print("=" * 50)
    print("Base RL Model Training v2")
    print("=" * 50)
    print(f"Dataset: {len(dataset)} fonksiyon")
    print()
    
    # Base model eğit - Efficiency-based reward ile optimize edildi
    # Early stopping ve duplicate penalty sayesinde daha az step gerekli
    train_base_model(
        dataset, 
        timesteps_per_code=30000,  # Azaltıldı: 50k -> 30k (efficiency sayesinde)
        model_path="ppo_testgen_base.zip",
        checkpoint_freq=10000
    )
    
    print("\n" + "=" * 50)
    print("Eğitim tamamlandı!")
    print("=" * 50)
    print("\nKullanım:")
    print("1. Arayüzü başlat: streamlit run main.py")
    print("2. veya: from rl_module_v2 import generate_test_cases")

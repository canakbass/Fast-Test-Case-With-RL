"""
Gerçekçi ve çeşitli Python fonksiyonlarından oluşan büyük bir dataset örneği.
Gerçek projelerden, open-source kodlardan, algoritma örneklerinden alınabilir.
"""
dataset = [
    ("""
def factorial(n):
    if n < 0:
        raise ValueError('Negative number')
    if n == 0:
        return 1
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
""", "factorial"),
    ("""
def is_prime(n):
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
""", "is_prime"),
    ("""
def gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)
""", "gcd"),
    ("""
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
""", "merge_sort"),
    ("""
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
""", "binary_search"),
    ("""
def count_vowels(s):
    vowels = 'aeiouAEIOU'
    count = 0
    for char in s:
        if char in vowels:
            count += 1
    return count
""", "count_vowels"),
    ("""
def reverse_string(s):
    return s[::-1]
""", "reverse_string"),
    ("""
def is_palindrome(s):
    s = s.lower().replace(' ', '')
    return s == s[::-1]
""", "is_palindrome"),
    ("""
def fizzbuzz(n):
    for i in range(1, n+1):
        if i % 15 == 0:
            print('FizzBuzz')
        elif i % 3 == 0:
            print('Fizz')
        elif i % 5 == 0:
            print('Buzz')
        else:
            print(i)
""", "fizzbuzz"),
    ("""
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
""", "quicksort"),
    # ...daha fazla fonksiyon eklenebilir...
]

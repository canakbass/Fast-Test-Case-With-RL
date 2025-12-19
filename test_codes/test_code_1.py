"""Test file 1: String operations"""


def reverse_string(s: str) -> str:
    """Reverse a string."""
    if not s:
        raise ValueError("Empty string")
    return s[::-1]


def is_palindrome(s: str) -> bool:
    """Check if string is palindrome."""
    if not s:
        return False
    s = s.lower()
    return s == s[::-1]


def count_vowels(s: str) -> int:
    """Count vowels in string."""
    vowels = "aeiouAEIOU"
    count = 0
    for char in s:
        if char in vowels:
            count += 1
    return count


def capitalize_words(s: str) -> str:
    """Capitalize first letter of each word."""
    if not s:
        return ""
    return s.title()

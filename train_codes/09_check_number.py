"""Training code 9: Check if number is positive, negative, or zero"""

def check_number(n: int):
    if n > 0:
        return "positive"
    elif n < 0:
        return "negative"
    else:
        return "zero"

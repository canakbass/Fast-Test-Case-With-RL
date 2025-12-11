
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

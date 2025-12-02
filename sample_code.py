def my_math_function(x, y, op):
    """
    A simple math function to test.
    op: 1 for add, 2 for sub, 3 for mul, 4 for div
    """
    if op == 1:
        return x + y
    elif op == 2:
        return x - y
    elif op == 3:
        return x * y
    elif op == 4:
        if y != 0:
            return x / y
        else:
            return 0
    else:
        return -1

import math
from typing import List, Union

class AdvancedCalculator:
    def __init__(self):
        self.history: List[str] = []

    def _add_to_history(self, operation: str, result: Union[int, float]) -> None:
        entry = f"{operation} = {result}"
        self.history.append(entry)

    def add(self, a: float, b: float) -> float:
        result = a + b
        self._add_to_history(f"{a} + {b}", result)
        return result

    def subtract(self, a: float, b: float) -> float:
        result = a - b
        self._add_to_history(f"{a} - {b}", result)
        return result

    def multiply(self, a: float, b: float) -> float:
        result = a * b
        self._add_to_history(f"{a} * {b}", result)
        return result

    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Error: Division by zero is not allowed.")
        result = a / b
        self._add_to_history(f"{a} / {b}", result)
        return result

    def modulo(self, a: float, b: float) -> float:
        result = a % b
        self._add_to_history(f"{a} % {b}", result)
        return result

    def power(self, base: float, exponent: float) -> float:
        result = math.pow(base, exponent)
        self._add_to_history(f"{base} ^ {exponent}", result)
        return result

    def square_root(self, a: float) -> float:
        if a < 0:
            raise ValueError("Error: Cannot calculate square root of a negative number.")
        result = math.sqrt(a)
        self._add_to_history(f"sqrt({a})", result)
        return result


    def logarithm(self, a: float, base: float = 10) -> float:
        if a <= 0 or base <= 0:
            raise ValueError("Error: Logarithm input and base must be positive.")
        result = math.log(a, base)
        self._add_to_history(f"log{base}({a})", result)
        return result

    def sin(self, a: float) -> float:
        result = math.sin(a)
        self._add_to_history(f"sin({a})", result)
        return result

    def cos(self, a: float) -> float:
        result = math.cos(a)
        self._add_to_history(f"cos({a})", result)
        return result

    def tan(self, a: float) -> float:
        result = math.tan(a)
        self._add_to_history(f"tan({a})", result)
        return result

    def get_history(self) -> List[str]:
        return self.history

    def clear_history(self) -> None:
        self.history = []
from llm_module import generate_pytest_code

# Basit test case'ler
test_cases = [
    {'function': 'add', 'input': [1.0, 2.0], 'output': 3.0, 'exception': None},
    {'function': 'add', 'input': [0.0, 0.0], 'output': 0.0, 'exception': None},
    {'function': 'divide', 'input': [10.0, 2.0], 'output': 5.0, 'exception': None},
    {'function': 'divide', 'input': [10.0, 0.0], 'output': None, 'exception': 'ValueError'},
]

code = '''
class Calculator:
    def add(self, a, b):
        return a + b
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Div by zero")
        return a / b
'''

pytest_code = generate_pytest_code(test_cases, code)
print(pytest_code)

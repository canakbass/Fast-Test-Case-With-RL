from llm_module import generate_pytest_code

code = '''
class Calculator:
    def add(self, a, b):
        return a + b
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError()
        return a / b
'''

# Fake test cases simulating RL output
test_cases = []
for i in range(10):
    test_cases.append({'function': 'add', 'input': [i, i+1], 'output': i*2+1})
for i in range(8):
    test_cases.append({'function': 'divide', 'input': [i+1, 2], 'output': (i+1)/2})
test_cases.append({'function': 'divide', 'input': [10, 0], 'exception': 'ValueError'})

print(f'Total cases: {len(test_cases)}')
add_count = sum(1 for c in test_cases if c['function'] == 'add')
div_count = sum(1 for c in test_cases if c['function'] == 'divide')
print(f'Functions: add ({add_count} cases), divide ({div_count} cases)')
print()

pytest_code = generate_pytest_code(test_cases, code)
print(pytest_code)

# Test sayısını say
test_count = pytest_code.count('def test_')
print(f'\n\n==> Generated {test_count} test functions')

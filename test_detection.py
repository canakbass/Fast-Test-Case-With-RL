"""Test function detection for test_code files"""
import rl_module_v2

# Test 1: String operations
code1 = open('test_code_1.py', encoding='utf-8').read()
functions1 = rl_module_v2.get_all_functions(code1)
print(f"test_code_1.py: {len(functions1)} fonksiyon")
for f in functions1:
    print(f"  - {f['name']} ({f['num_params']} parametre)")

# Test 2: Math operations  
code2 = open('test_code_2.py', encoding='utf-8').read()
functions2 = rl_module_v2.get_all_functions(code2)
print(f"\ntest_code_2.py: {len(functions2)} fonksiyon")
for f in functions2:
    print(f"  - {f['name']} ({f['num_params']} parametre)")

# Test 3: Data structures
code3 = open('test_code_3.py', encoding='utf-8').read()
functions3 = rl_module_v2.get_all_functions(code3)
print(f"\ntest_code_3.py: {len(functions3)} metod")
for f in functions3:
    cls = f"[{f['class_name']}]" if f['is_method'] else ""
    print(f"  - {cls} {f['name']} ({f['num_params']} parametre)")

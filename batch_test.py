"""
Batch test script - Test multiple code files
"""
from rl_module_v2 import generate_test_cases
from llm_module import generate_pytest_code
import os

test_files = [
    # 'test_code_1.py',  # String operations - SKIPPED (filtered by is_numeric_compatible)
    'test_code_2.py',  # Math operations
    'test_code_3.py',  # Data structures
]

print("="*60)
print("BATCH TEST CASE GENERATION")
print("="*60)

for i, filename in enumerate(test_files, 1):
    print(f"\n[{i}/{len(test_files)}] Processing: {filename}")
    print("-"*60)
    
    # Kodu oku
    with open(filename, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Test case'leri üret (az episode ile hızlı test)
    cases, info = generate_test_cases(code, num_episodes=2)
    
    print(f"Generated {len(cases)} test cases")
    print(f"Functions tested: {len(info['functions'])}")
    for func_info in info['functions']:
        print(f"  - {func_info['name']}: {func_info['cases']} cases, "
              f"{func_info['coverage_pct']:.1f}% coverage")
    
    # Pytest koduna çevir
    pytest_code = generate_pytest_code(cases, code)
    test_count = pytest_code.count('def test_')
    
    # Dosyaya kaydet
    output_file = f"test_generated_{filename}"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(pytest_code)
    
    print(f"✓ Generated {test_count} pytest functions")
    print(f"✓ Saved to: {output_file}")

print("\n" + "="*60)
print("BATCH TEST GENERATION COMPLETE")
print("="*60)

import rl_module_v2

code = open('test_code_2.py', encoding='utf-8').read()
cases, info = rl_module_v2.generate_test_cases(code, num_episodes=2)

print(f'\n{"="*60}')
print(f'SONUÇ: {len(cases)} test case')
print(f'Coverage: {info["coverage_lines"]}/{info["total_lines"]} lines')
print(f'Percentage: {info["coverage_pct"]:.1f}%')
print(f'Exceptions: {info["exceptions_found"]}')
print(f'{"="*60}')

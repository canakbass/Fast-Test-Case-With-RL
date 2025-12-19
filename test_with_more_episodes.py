"""
Test model behavior with more episodes
See if efficiency reward helps with more episodes
"""
import rl_module_v2

code = open('test_code_2.py', encoding='utf-8').read()

print("="*70)
print("TESTING WITH 5 EPISODES (more realistic scenario)")
print("="*70)

print("\n[1/2] NEW efficiency-trained model...")
print("-"*70)
cases_new, info_new = rl_module_v2.generate_test_cases(
    code, 
    model_path="ppo_testgen_base.zip",
    num_episodes=5
)

print("\n[2/2] OLD model...")
print("-"*70)
cases_old, info_old = rl_module_v2.generate_test_cases(
    code, 
    model_path="ppo_testgen_base_OLD_before_efficiency.zip",
    num_episodes=5
)

print("\n" + "="*70)
print("RESULTS (5 episodes)")
print("="*70)
print(f"Old Model: {len(cases_old)} tests, {info_old['coverage_pct']:.1f}% coverage")
print(f"New Model: {len(cases_new)} tests, {info_new['coverage_pct']:.1f}% coverage")

reduction = (1 - len(cases_new)/len(cases_old)) * 100
if reduction > 0:
    print(f"\n✓ {reduction:.1f}% fewer tests with same/better coverage!")
elif reduction < 0:
    print(f"\n⚠ {abs(reduction):.1f}% more tests (but still good coverage)")
else:
    print(f"\n= Same number of tests")

efficiency_old = info_old['coverage_pct'] / len(cases_old) * 100
efficiency_new = info_new['coverage_pct'] / len(cases_new) * 100
print(f"\nEfficiency (coverage/test):")
print(f"  Old: {efficiency_old:.3f}")
print(f"  New: {efficiency_new:.3f} ({(efficiency_new/efficiency_old-1)*100:+.1f}%)")

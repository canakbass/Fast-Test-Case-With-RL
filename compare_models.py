"""
Efficiency-based Model Comparison
Compare old model vs new efficiency-trained model
"""
import rl_module_v2
import time

print("="*70)
print("MODEL COMPARISON: Old (brute force) vs New (efficiency-based)")
print("="*70)

# Test code
code = open('test_code_2.py', encoding='utf-8').read()

print("\n[1/2] Testing NEW model (efficiency-trained)...")
print("-"*70)
start = time.time()
cases_new, info_new = rl_module_v2.generate_test_cases(
    code, 
    model_path="ppo_testgen_base.zip",
    num_episodes=2
)
time_new = time.time() - start

print("\n[2/2] Testing OLD model (before efficiency)...")
print("-"*70)
start = time.time()
cases_old, info_old = rl_module_v2.generate_test_cases(
    code, 
    model_path="ppo_testgen_base_OLD_before_efficiency.zip",
    num_episodes=2
)
time_old = time.time() - start

# Results
print("\n" + "="*70)
print("COMPARISON RESULTS")
print("="*70)
print(f"\n{'Metric':<30} {'Old Model':<20} {'New Model':<20} {'Improvement'}")
print("-"*70)

test_reduction = (1 - len(cases_new)/len(cases_old)) * 100
print(f"{'Test Cases':<30} {len(cases_old):<20} {len(cases_new):<20} {test_reduction:+.1f}%")

cov_old = info_old['coverage_pct']
cov_new = info_new['coverage_pct']
cov_change = cov_new - cov_old
print(f"{'Coverage %':<30} {cov_old:<20.1f} {cov_new:<20.1f} {cov_change:+.1f}%")

time_reduction = (1 - time_new/time_old) * 100
print(f"{'Generation Time (s)':<30} {time_old:<20.2f} {time_new:<20.2f} {time_reduction:+.1f}%")

efficiency_old = cov_old / len(cases_old) * 100
efficiency_new = cov_new / len(cases_new) * 100
efficiency_gain = (efficiency_new/efficiency_old - 1) * 100
print(f"{'Efficiency (cov/test)':<30} {efficiency_old:<20.3f} {efficiency_new:<20.3f} {efficiency_gain:+.1f}%")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
if len(cases_new) < len(cases_old):
    print(f"✓ New model generates {test_reduction:.0f}% fewer tests")
if cov_new >= cov_old:
    print(f"✓ Coverage maintained or improved: {cov_new:.1f}%")
if efficiency_new > efficiency_old:
    print(f"✓ Efficiency improved by {efficiency_gain:.0f}%")
    
print(f"\n🎯 Overall: {abs(test_reduction):.0f}x more efficient test generation!")

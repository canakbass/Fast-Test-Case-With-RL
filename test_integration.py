"""
Boundary Analysis + RL Entegrasyon Testi
"""
from rl_module_v2 import TestCaseGeneratorEnv, train_base_model, generate_test_cases
from boundary_analysis import extract_boundary_values, analyze_code_for_testing, calculate_boundary_reward

# Test kodu
test_code = '''
def factorial(n: int):
    """Faktöriyel hesapla"""
    if n < 0:
        raise ValueError("Negative number not allowed")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def is_prime(n: int):
    """Asal sayı kontrolü"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True
'''

print("=" * 60)
print("BOUNDARY ANALYSIS + RL ENTEGRASYON TESTİ")
print("=" * 60)

# 1. Boundary Analysis
print("\n[1] BOUNDARY VALUE ANALYSIS")
print("-" * 40)
analysis = analyze_code_for_testing(test_code)

print("Fonksiyonlar:")
for func in analysis['functions']:
    print(f"  • {func['name']}: {len(func['params'])} param, {func['branches']} branch")
    if func['has_error_handling']:
        print(f"    ⚠️ Exception handling var")

print("\nBulunan Koşullar:")
for cond in analysis['boundary_analysis']['conditions']:
    print(f"  • {cond}")

print("\nKritik Test Değerleri:")
print(f"  {analysis['boundary_analysis']['critical_values']}")

# 2. Environment Test
print("\n[2] RL ENVIRONMENT TEST")
print("-" * 40)
env = TestCaseGeneratorEnv()
env.set_code(test_code, "test_module")

print(f"Code features shape: {env.code_features.shape}")
print(f"Observation space: {env.observation_space.shape}")
print(f"Action space: {env.action_space.shape}")
print(f"Loaded critical values: {env.critical_values}")
print(f"Number of params: {len(env.current_params)}")

# 3. Reward Test
print("\n[3] BOUNDARY REWARD TEST")
print("-" * 40)
test_actions = [
    [0, 0, 0, 0, 0],      # Kritik değer: 0
    [1, 1, 1, 1, 1],      # Kritik değer: 1
    [-1, 0, 1, 2, 3],     # Karışık kritik değerler
    [50, 50, 50, 50, 50], # Normal değer (kritik değil)
]

for action in test_actions:
    reward = calculate_boundary_reward(action, test_code)
    print(f"  Action {action[:3]}... → Boundary Reward: {reward}")

# 4. Step Test
print("\n[4] ENVIRONMENT STEP TEST")
print("-" * 40)
obs, _ = env.reset()
print(f"Initial observation shape: {obs.shape}")

# Birkaç step at
for i, action in enumerate(test_actions[:2]):
    action_arr = [float(x) for x in action]
    obs, reward, term, trunc, info = env.step(action_arr)
    print(f"Step {i+1}: action={action[:2]}, reward={reward}, lines_covered={len(env.covered_lines)}")

print(f"\nUseful cases found: {len(env.useful_cases)}")
for case in env.useful_cases[:3]:
    print(f"  • Input: {case['input']}, Exception: {case.get('exception', 'None')}")

print("\n" + "=" * 60)
print("✅ ENTEGRASYON TESTİ TAMAMLANDI")
print("=" * 60)

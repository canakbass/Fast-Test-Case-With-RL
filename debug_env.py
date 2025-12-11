"""
Debug: RL environment trace test
"""

import sys
sys.path.insert(0, '.')

from rl_module_v2 import TestCaseGeneratorEnv
import numpy as np

# Test kodu
test_code = """
def add(a: int, b: int):
    if a < 0:
        return -1
    return a + b
"""

print("=== RL Environment Debug ===\n")

# Environment oluştur
env = TestCaseGeneratorEnv()
env.set_code(test_code, "test_add")

print(f"Function: {env.current_func}")
print(f"Params: {env.current_params}")
print(f"Total lines (executable): {env.total_lines}")

# Reset
obs, info = env.reset()
print(f"\nObservation shape: {obs.shape}")

# Manuel step
print("\n--- Step 1: action=[5.0, 3.0, 0, 0, 0] ---")
action = np.array([5.0, 3.0, 0, 0, 0], dtype=np.float32)
obs, reward, terminated, truncated, info = env.step(action)

print(f"Reward: {reward}")
print(f"Covered lines: {env.covered_lines}")
print(f"Useful cases: {len(env.useful_cases)}")

# Step 2 - negatif değer
print("\n--- Step 2: action=[-5.0, 3.0, 0, 0, 0] ---")
action2 = np.array([-5.0, 3.0, 0, 0, 0], dtype=np.float32)
obs, reward, terminated, truncated, info = env.step(action2)

print(f"Reward: {reward}")
print(f"Covered lines: {env.covered_lines}")
print(f"Useful cases: {len(env.useful_cases)}")

# Step 3 - sıfır değer
print("\n--- Step 3: action=[0, 0, 0, 0, 0] ---")
action3 = np.array([0, 0, 0, 0, 0], dtype=np.float32)
obs, reward, terminated, truncated, info = env.step(action3)

print(f"Reward: {reward}")
print(f"Covered lines: {env.covered_lines}")
print(f"Useful cases: {len(env.useful_cases)}")

print("\n✓ Debug tamamlandı!")

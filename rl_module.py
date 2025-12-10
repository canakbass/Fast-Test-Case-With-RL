import gymnasium as gym
from gymnasium import spaces
import numpy as np
import coverage
import inspect
import importlib.util
import os
import torch
import radon.raw as radon_raw
from tqdm import trange
from stable_baselines3 import PPO

# --- Helper Functions ---

def save_model(model, path="ppo_model.zip"):
    model.save(path)

def load_model(env, path="ppo_model.zip"):
    from stable_baselines3 import PPO
    return PPO.load(path, env=env)

def fine_tune_model(model, env, timesteps=1000):
    model.set_env(env)
    model.learn(total_timesteps=timesteps)
    return model

# --- Environment Class ---

class TestGenEnv(gym.Env):
    def __init__(self, func_to_test, source_file):
        super(TestGenEnv, self).__init__()
        self.func = func_to_test
        self.source_file = source_file

        # Determine number of arguments
        sig = inspect.signature(self.func)
        self.num_args = len(sig.parameters)

        self.action_space = spaces.Box(low=-100, high=100, shape=(self.num_args,), dtype=np.float32)

        with open(source_file, 'r') as f:
            self.lines = f.readlines()
        self.num_lines = len(self.lines)

        self.max_lines = 1000
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.max_lines,), dtype=np.int8)

        self.covered_lines = set()
        self.current_coverage_vec = np.zeros(self.max_lines, dtype=np.int8)

        self.useful_cases = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.covered_lines = set()
        self.current_coverage_vec = np.zeros(self.max_lines, dtype=np.int8)
        return self.current_coverage_vec, {}

    def step(self, action):
        # We start coverage WITHOUT source argument to trace everything, then filter.
        # This avoids "Module never imported" warning if coverage can't map source file to module.
        cov = coverage.Coverage(data_file=None)
        cov.start()
        try:
            args = []
            params = list(inspect.signature(self.func).parameters.values())
            for i, p in enumerate(params):
                if p.annotation == int:
                    args.append(int(action[i]))
                else:
                    args.append(action[i])

            self.func(*args)
        except Exception as e:
            pass
        cov.stop()

        data = cov.get_data()

        abs_path = os.path.abspath(self.source_file)
        executed_lines = data.lines(abs_path)

        # If absolute path didn't work, try realpath (handling symlinks)
        if executed_lines is None:
            executed_lines = data.lines(os.path.realpath(self.source_file))

        # If still None, try base filename if it matches (risky but maybe needed)
        # But data.lines() expects a file path that matches what coverage recorded.
        # Coverage records absolute paths usually.

        new_lines = set()
        if executed_lines:
            new_lines = set(executed_lines)

        reward = len(new_lines - self.covered_lines)

        if reward > 0:
            self.useful_cases.append({"input": [float(x) for x in action], "new_lines": list(new_lines - self.covered_lines)})

        self.covered_lines.update(new_lines)

        for line in self.covered_lines:
            if 0 <= line - 1 < self.max_lines:
                self.current_coverage_vec[line - 1] = 1

        terminated = False
        truncated = False

        return self.current_coverage_vec, reward, terminated, truncated, {}

# --- Main Logic Functions ---

def train_base_model(dataset, timesteps=2000000, model_path="ppo_base_model.zip", checkpoint_freq=50000):
    for code_content, module_name in dataset:
        file_path = f"{module_name}.py"
        with open(file_path, "w") as f:
            f.write(code_content)
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target_func = None
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and obj.__module__ == module_name:
                target_func = obj
                break
        if not target_func:
            continue
        env = TestGenEnv(target_func, file_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        checkpoint_path = f"{model_path.replace('.zip', '')}_checkpoint_{module_name}.zip"
        if os.path.exists(checkpoint_path):
            print(f"Checkpoint bulundu: {checkpoint_path}, devam ediliyor...")
            model = load_model(env, checkpoint_path)
        else:
            model = PPO("MlpPolicy", env, verbose=0, device=device)

        steps_per_iter = 10000
        total_steps = 0
        try:
            for i in trange(0, timesteps, steps_per_iter, desc=f"{module_name} RL Training", unit="step"):
                model.learn(total_timesteps=min(steps_per_iter, timesteps-i), reset_num_timesteps=False)
                total_steps += min(steps_per_iter, timesteps-i)
                if total_steps % checkpoint_freq == 0:
                    save_model(model, checkpoint_path)
                    print(f"\nCheckpoint kaydedildi: {checkpoint_path} ({total_steps}/{timesteps} steps)")
        except KeyboardInterrupt:
            print(f"\nEğitim kesintiye uğradı! Son checkpoint kaydediliyor: {checkpoint_path}")
            save_model(model, checkpoint_path)
            print(f"Checkpoint kaydedildi. Devam etmek için aynı scripti tekrar çalıştırın.")
            raise

        save_model(model, model_path)
        print(f"\nEğitim tamamlandı! Final model: {model_path}")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
    return model_path

def iterative_fine_tune(code_content, module_name="temp_module", base_model_path="ppo_base_model.zip", fine_tune_steps=20000, coverage_threshold=85, max_rounds=3):
    file_path = f"{module_name}.py"
    with open(file_path, "w") as f:
        f.write(code_content)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target_func = None
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and obj.__module__ == module_name:
            target_func = obj
            break
    if not target_func:
        return [], 0, 0
    env = TestGenEnv(target_func, file_path)

    if os.path.exists(base_model_path):
        model = load_model(env, base_model_path)
    else:
        print(f"Base model {base_model_path} not found. Training from scratch.")
        model = PPO("MlpPolicy", env, verbose=1)

    cases = []
    coverage = 0
    for round in range(max_rounds):
        model = fine_tune_model(model, env, timesteps=fine_tune_steps)
        # Note: we are NOT saving back to base_model_path to avoid overwriting general model with specific one
        # unless that is desired. The original code did it.
        # But wait, if we don't save, subsequent rounds use the model in memory.
        # The loop uses `model` object so it carries over.

        actual_env = model.env.envs[0]
        while hasattr(actual_env, 'env'):
            if isinstance(actual_env, TestGenEnv):
                break
            actual_env = actual_env.env
        if isinstance(actual_env, TestGenEnv):
            cases = actual_env.useful_cases
            # Calculate coverage from cases found so far

            all_covered_lines = set()
            for c in cases:
                all_covered_lines.update(c['new_lines'])

            # Total executable lines
            cov = gym_coverage_analysis(file_path)
            total_statements = cov['total']
            if total_statements > 0:
                coverage = (len(all_covered_lines) / total_statements) * 100
            else:
                coverage = 0

        if coverage >= coverage_threshold:
            break
    return cases, coverage, round+1

def gym_coverage_analysis(code_path):
    # Helper to get total statements
    cov = coverage.Coverage(source=[code_path], data_file=None)
    cov.start()
    cov.stop()
    analysis = cov.analysis(code_path)
    # analysis: (filename, statements, excluded, missing, missing_formatted)
    return {'total': len(analysis[1]), 'statements': analysis[1]}


def train_and_generate_cases(code_content, module_name="temp_module", timesteps=1000, mode="train", model_path="ppo_model.zip", fine_tune_steps=500):
    file_path = f"{module_name}.py"
    with open(file_path, "w") as f:
        f.write(code_content)

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    target_func = None
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and obj.__module__ == module_name:
            target_func = obj
            break

    if not target_func:
        return []

    env = TestGenEnv(target_func, file_path)

    if mode == "train":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        model = PPO("MlpPolicy", env, verbose=1, device=device)
        model.learn(total_timesteps=timesteps)
        save_model(model, model_path)
    elif mode == "load":
        model = load_model(env, model_path)
    elif mode == "fine_tune":
        model = load_model(env, model_path)
        model = fine_tune_model(model, env, timesteps=fine_tune_steps)
        save_model(model, model_path)
    else:
        raise ValueError("Unknown mode")

    actual_env = model.env.envs[0]
    while hasattr(actual_env, 'env'):
        if isinstance(actual_env, TestGenEnv):
            break
        actual_env = actual_env.env

    if isinstance(actual_env, TestGenEnv):
        return actual_env.useful_cases
    else:
        return []

if __name__ == "__main__":
    sample_code = """
def calculator(a: int, b: int):
    if a > 0:
        if b > 0:
            return a + b
        else:
            return a - b
    else:
        return a * b
    """
    cases = train_and_generate_cases(sample_code, timesteps=500)
    print("Generated cases:", cases)

import coverage
import radon.raw as radon_raw
from tqdm import trange
import torch
# --- Coverage Ölçüm Fonksiyonu ---
def measure_coverage(code_path, func_name=None):
    cov = coverage.Coverage(source=[code_path], data_file=None)
    cov.start()
    # Fonksiyonun tüm branch'lerini tetiklemek için dummy inputlar ile çalıştırmak gerekebilir
    # Burada sadece dosya coverage'ı ölçülür
    cov.stop()
    analysis = cov.analysis(code_path)
    executed = set(analysis[1])
    missing = set(analysis[3])
    total = len(executed) + len(missing)
    percent = 100 * len(executed) / total if total > 0 else 0
    return percent
def train_base_model(dataset, timesteps=2000000, model_path="ppo_base_model.zip", checkpoint_freq=50000):
    """
    dataset: List of (code_content, module_name)
    Uzun süreli base RL eğitimi. Her kod için ayrı env ile PPO eğitimi.
    checkpoint_freq: Her kaç adımda bir checkpoint kaydedileceği
    """
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
        # Eğer checkpoint varsa devam et
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
                # Her checkpoint_freq adımda bir kaydet
                if total_steps % checkpoint_freq == 0:
                    save_model(model, checkpoint_path)
                    print(f"\nCheckpoint kaydedildi: {checkpoint_path} ({total_steps}/{timesteps} steps)")
        except KeyboardInterrupt:
            print(f"\nEğitim kesintiye uğradı! Son checkpoint kaydediliyor: {checkpoint_path}")
            save_model(model, checkpoint_path)
            print(f"Checkpoint kaydedildi. Devam etmek için aynı scripti tekrar çalıştırın.")
            raise
        
        # Eğitim tamamlandı, final model kaydet
        save_model(model, model_path)
        print(f"\nEğitim tamamlandı! Final model: {model_path}")
        # Checkpoint dosyasını sil (artık gerekli değil)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
    return model_path
def iterative_fine_tune(code_content, module_name="temp_module", base_model_path="ppo_base_model.zip", fine_tune_steps=20000, coverage_threshold=85, max_rounds=3):
    """
    Base model ile test case üret, coverage düşükse max 3 döngü fine-tune et.
    """
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
    model = load_model(env, base_model_path)
    cases = []
    coverage = 0
    for round in range(max_rounds):
        model = fine_tune_model(model, env, timesteps=fine_tune_steps)
        save_model(model, base_model_path)
        actual_env = model.env.envs[0]
        while hasattr(actual_env, 'env'):
            if isinstance(actual_env, TestGenEnv):
                break
            actual_env = actual_env.env
        if isinstance(actual_env, TestGenEnv):
            cases = actual_env.useful_cases
        coverage = measure_coverage(file_path)
        if coverage >= coverage_threshold:
            break
    return cases, coverage, round+1
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import coverage
import inspect
import importlib.util
import os
from stable_baselines3 import PPO

def save_model(model, path="ppo_model.zip"):
    model.save(path)

def load_model(env, path="ppo_model.zip"):
    from stable_baselines3 import PPO
    return PPO.load(path, env=env)

def fine_tune_model(model, env, timesteps=1000):
    model.set_env(env)
    model.learn(total_timesteps=timesteps)
    return model

class TestGenEnv(gym.Env):
    def __init__(self, func_to_test, source_file):
        super(TestGenEnv, self).__init__()
        self.func = func_to_test
        self.source_file = source_file

        # Determine number of arguments
        sig = inspect.signature(self.func)
        self.num_args = len(sig.parameters)

        # Action space: generating inputs. Assuming float inputs for simplicity.
        # Can be adjusted based on type hints if available.
        # We use a continuous space [-100, 100]
        self.action_space = spaces.Box(low=-100, high=100, shape=(self.num_args,), dtype=np.float32)

        # State space: Coverage representation.
        # For simplicity, let's use a binary vector representing executed lines in the file.
        # We need to know the number of lines in the file.
        with open(source_file, 'r') as f:
            self.lines = f.readlines()
        self.num_lines = len(self.lines)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_lines,), dtype=np.int8)

        self.covered_lines = set()
        self.current_coverage_vec = np.zeros(self.num_lines, dtype=np.int8)

        # Keep track of useful test cases
        self.useful_cases = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.covered_lines = set()
        self.current_coverage_vec = np.zeros(self.num_lines, dtype=np.int8)
        # self.useful_cases = [] # Do not reset useful cases if we want to collect them over episodes,
                                 # but usually reset clears state. We can collect them externally or in a separate list.
        return self.current_coverage_vec, {}

    def step(self, action):
        # Action is a list of arguments
        # We need to convert them to what the function expects if possible
        # For now, pass as is (floats).

        # Run function with coverage
        # We need to make sure coverage knows about the file.
        # Use data_file=None for in-memory coverage (faster)
        cov = coverage.Coverage(source=[self.source_file], data_file=None)
        cov.start()
        try:
            # We assume the function handles numbers.
            # If it expects int, we cast.
            # Heuristic: if action is close to int, cast to int?
            # Or check type hints.
            args = []
            params = list(inspect.signature(self.func).parameters.values())
            for i, p in enumerate(params):
                if p.annotation == int:
                    args.append(int(action[i]))
                else:
                    args.append(action[i])

            self.func(*args)
        except Exception as e:
            # Function crashed, might be a bug or invalid input.
            # We can treat this as "no coverage increase" or small penalty?
            # Or maybe finding a crash is a reward?
            pass
        cov.stop()

        # Analyze coverage
        # cov.analysis(source_file) returns 5 elements: (filename, statements, excluded, missing, missing_formatted)
        # But we just need to know which lines were actually executed in this run.

        data = cov.get_data()

        # data.lines(abspath) returns the list of executed line numbers.
        # We need absolute path usually.
        abs_path = os.path.abspath(self.source_file)
        # Coverage might use a relative path if that's how it was started.
        # Try both.
        executed_lines = data.lines(abs_path)
        if executed_lines is None:
             executed_lines = data.lines(self.source_file)

        new_lines = set()
        if executed_lines:
            new_lines = set(executed_lines)

        # Calculate reward: number of newly covered lines
        reward = len(new_lines - self.covered_lines)

        if reward > 0:
            self.useful_cases.append({"input": [float(x) for x in action], "new_lines": list(new_lines - self.covered_lines)})

        self.covered_lines.update(new_lines)

        # Update state
        for line in self.covered_lines:
            if 0 <= line - 1 < self.num_lines:
                self.current_coverage_vec[line - 1] = 1

        # Done condition?
        # Maybe when coverage is 100% or after fixed steps (handled by TimeLimit wrapper usually)
        # For now, we continue until max steps of the episode
        terminated = False
        truncated = False

        return self.current_coverage_vec, reward, terminated, truncated, {}

def train_and_generate_cases(code_content, module_name="temp_module", timesteps=1000, mode="train", model_path="ppo_model.zip", fine_tune_steps=500):
    """
    mode: "train" -> yeni model eğit ve kaydet
          "load"  -> hazır modeli yükle ve test case üret
          "fine_tune" -> hazır modeli fine-tune et ve test case üret
    """
    # Save code to file
    file_path = f"{module_name}.py"
    with open(file_path, "w") as f:
        f.write(code_content)

    # Import the module
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the first function to test
    target_func = None
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and obj.__module__ == module_name:
            target_func = obj
            break

    if not target_func:
        return []

    # Create Env
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

    # Return collected useful cases
    actual_env = model.env.envs[0]
    while hasattr(actual_env, 'env'):
        if isinstance(actual_env, TestGenEnv):
            break
        actual_env = actual_env.env

    if isinstance(actual_env, TestGenEnv):
        return actual_env.useful_cases
    else:
        return []
    # Save code to file
    file_path = f"{module_name}.py"
    with open(file_path, "w") as f:
        f.write(code_content)

    # Import the module
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the first function to test
    target_func = None
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and obj.__module__ == module_name:
            target_func = obj
            break

    if not target_func:
        return []

    # Create Env
    env = TestGenEnv(target_func, file_path)

    # Train Agent
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)

    # Return collected useful cases
    # Note: env might be re-instantiated or reset, so accessing env.useful_cases directly from the passed instance
    # works if DummyVecEnv is not wrapping it in a way that hides it.
    # Stable baselines wraps env. We can access original env via model.get_env().envs[0] if using DummyVecEnv

    # A safer way is to use a Callback to collect data, but for simplicity let's try accessing the env.

    # Since we passed the env instance directly to PPO (which wraps it), we can try:
    actual_env = model.env.envs[0]
    # Check if it's our env (might be wrapped in Monitor etc)
    while hasattr(actual_env, 'env'):
        if isinstance(actual_env, TestGenEnv):
            break
        actual_env = actual_env.env

    if isinstance(actual_env, TestGenEnv):
        return actual_env.useful_cases
    else:
        return []

if __name__ == "__main__":
    # Test logic
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

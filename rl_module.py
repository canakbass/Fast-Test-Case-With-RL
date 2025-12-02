import gymnasium as gym
from gymnasium import spaces
import numpy as np
import coverage
import inspect
import importlib.util
import os
from stable_baselines3 import PPO

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

def train_and_generate_cases(code_content, module_name="temp_module", timesteps=1000):
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

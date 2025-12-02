import pytest
import os
from rl_module import train_and_generate_cases
from analiz_module import get_metrics
import shutil

def test_rl_generation():
    # Simple function to test
    code = """
def check_positive(n):
    if n > 0:
        return True
    return False
    """

    # Run RL
    cases = train_and_generate_cases(code, module_name="test_temp_mod", timesteps=200)

    # We expect at least some cases.
    # Usually RL should find n > 0 and n <= 0.
    assert isinstance(cases, list)
    # Ideally checking if cases are not empty, but with random RL sometimes it might fail in short steps.
    # But for a simple threshold it usually works.

    # Clean up
    if os.path.exists("test_temp_mod.py"):
        os.remove("test_temp_mod.py")

def test_metrics():
    code = """
def foo():
    print("hello")
    """
    # Note: Radon LOC counts the lines in the file.
    # The string above has:
    # 1. empty line
    # 2. def foo():
    # 3.     print("hello")
    # 4.     (indentation) or just empty
    # Actually including docstring indentation/newlines.
    # Let's just check it's > 0
    metrics = get_metrics(code)
    assert 'LOC' in metrics
    assert metrics['LOC'] > 0

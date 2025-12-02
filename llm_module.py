import openai
import os

def generate_pytest_code(test_cases, code_content):
    """
    Uses OpenAI API to convert test cases into pytest code.
    test_cases: List of dicts with inputs.
    code_content: The original source code being tested.
    """

    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        return _generate_mock_pytest_code(test_cases)

    client = openai.OpenAI(api_key=api_key)

    prompt = f"""
    You are an expert Python tester. I have a Python function and a list of test inputs that were generated to maximize coverage.
    Your task is to write a comprehensive `pytest` test file for this code.

    Original Code:
    ```python
    {code_content}
    ```

    Generated Inputs (and covered lines):
    {test_cases}

    Please write a valid `test_generated.py` file that imports the function (assume the file is named `tested_module.py`) and runs these tests.
    Add assertions where possible (infer expected output or just check it runs without error if output is unknown).
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that writes python tests."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI: {e}")
        return _generate_mock_pytest_code(test_cases)

def _generate_mock_pytest_code(test_cases):
    """
    Fallback function to generate pytest code without LLM.
    """
    code = "import pytest\nfrom tested_module import *\n\n"

    for i, case in enumerate(test_cases):
        inputs = case['input']
        # formatting args
        args_str = ", ".join(map(str, inputs))

        code += f"def test_case_{i}():\n"
        code += f"    # Generated input: {inputs}\n"
        code += f"    # Targeted new lines: {case.get('new_lines')}\n"
        # Since we don't know the function name easily here without parsing again or passing it,
        # we will rely on the user to check valid code or the main loop to provide function name.
        # But wait, in the prompt to LLM we passed the code so LLM knows.
        # Here in fallback, we might struggle.
        # Let's assume the user knows what function to call or we assume a generic call if we can't parse.

        # Improvement: We should pass function name to this module too.
        # But for now, let's just create a comment saying "Call your function here"
        # or try to run the first function found in the code?

        code += f"    # TODO: Add assertion\n"
        code += f"    # func({args_str})\n"
        code += f"    pass\n\n"

    return code

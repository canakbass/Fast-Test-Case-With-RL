import openai
import os
import ast
import re

def extract_function_name(code_content):
    """Koddan ilk fonksiyon adını çıkar"""
    try:
        tree = ast.parse(code_content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node.name
    except:
        pass
    return "func"

def generate_pytest_code(test_cases, code_content):
    """
    Uses OpenAI API to convert test cases into pytest code.
    test_cases: List of dicts with inputs and 'function' field (class.method or function_name).
    code_content: The original source code being tested.
    """

    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        return _generate_mock_pytest_code(test_cases, code_content)

    client = openai.OpenAI(api_key=api_key)

    # Test case'leri fonksiyonlara göre grupla
    cases_by_function = {}
    for tc in test_cases:
        func_name = tc.get('function', 'unknown')
        if func_name not in cases_by_function:
            cases_by_function[func_name] = []
        cases_by_function[func_name].append(tc)

    # Her fonksiyon için ÖZET oluştur (LLM'e gönderilecek)
    function_summary = {}
    for func_name, cases in cases_by_function.items():
        # Örnek case'ler (max 5)
        sample_cases = []
        for i, tc in enumerate(cases[:5]):
            sample_cases.append({
                'inputs': tc['input'],
                'output': tc.get('output'),
                'exception': tc.get('exception')
            })
        
        function_summary[func_name] = {
            'total_cases': len(cases),
            'sample_cases': sample_cases,
            'exceptions_found': list(set(c.get('exception') for c in cases if c.get('exception')))
        }

    prompt = f"""
You are an expert Python tester. I have Python code with multiple functions/methods and test inputs generated using RL.

Original Code:
```python
{code_content}
```

Test Case Summary (by function):
{function_summary}

CRITICAL REQUIREMENTS:
1. Write tests for ALL {len(cases_by_function)} functions listed above
2. For each function, generate AT LEAST 5-10 diverse test cases (use the total_cases count as guidance)
3. For CLASS METHODS: Create a test class with setup_method that instantiates the class
4. For STANDALONE FUNCTIONS: Import and test directly
5. Include both normal cases AND exception tests
6. Use descriptive test names: test_<function>_<scenario>

Example structure for class:
```python
from module import AdvancedCalculator

class TestAdvancedCalculator:
    def setup_method(self):
        self.calc = AdvancedCalculator()
    
    def test_add_positive_numbers(self):
        assert self.calc.add(2.0, 3.0) == 5.0
    
    def test_divide_by_zero_raises_error(self):
        with pytest.raises(ValueError):
            self.calc.divide(10.0, 0.0)
```

Generate comprehensive pytest code for ALL functions. Output ONLY Python code.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that writes python tests. Output only valid Python code."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        result = response.choices[0].message.content
        
        # Markdown code block'larını temizle
        result = re.sub(r'^```python\n?', '', result)
        result = re.sub(r'\n?```$', '', result)
        result = result.strip()
        
        return result
    except Exception as e:
        print(f"Error calling OpenAI: {e}")
        return _generate_mock_pytest_code(test_cases, code_content)

def _generate_mock_pytest_code(test_cases, code_content):
    """
    Fallback function to generate pytest code without LLM.
    Handles both standalone functions and class methods.
    Generates representative tests (max 15 per function to avoid bloat).
    """
    # Test case'leri fonksiyonlara göre grupla
    cases_by_function = {}
    for tc in test_cases:
        func_name = tc.get('function', 'unknown')
        if func_name not in cases_by_function:
            cases_by_function[func_name] = []
        cases_by_function[func_name].append(tc)
    
    # Class'ları ve method mapping'lerini bul
    classes = {}  # {class_name: [method_names]}
    try:
        tree = ast.parse(code_content)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes[node.name] = [
                    item.name for item in node.body 
                    if isinstance(item, ast.FunctionDef) and not item.name.startswith('_')
                ]
    except:
        pass
    
    # Her fonksiyonun hangi class'a ait olduğunu bul
    func_to_class = {}
    for class_name, methods in classes.items():
        for method in methods:
            func_to_class[method] = class_name
    
    code = '''"""
Auto-generated test cases using RL-based Test Generator
"""
import pytest
'''
    
    if classes:
        # Import all classes
        all_classes = list(classes.keys())
        code += f"from target_code import {', '.join(all_classes)}\n\n"
        
        # Generate test class for each class
        for class_name, methods in classes.items():
            code += f"class Test{class_name}:\n"
            code += f"    def setup_method(self):\n"
            code += f"        self.instance = {class_name}()\n\n"
            
            # Filter test cases for this class's methods
            for method_name in methods:
                if method_name not in cases_by_function:
                    continue
                    
                cases = cases_by_function[method_name]
                
                # Her fonksiyon için MAX 10 normal + 5 exception test
                normal_cases = [c for c in cases if not c.get('exception')][:10]
                exception_cases = [c for c in cases if c.get('exception')][:5]
                
                total = len(normal_cases) + len(exception_cases)
                code += f"    # ===== {method_name}: {total} tests (from {len(cases)} generated cases) =====\n"
                
                for i, case in enumerate(normal_cases):
                    inputs = case['input']
                    args_str = ", ".join(str(x) for x in inputs)
                    output = case.get('output')
                    
                    code += f"    def test_{method_name}_case_{i+1}(self):\n"
                    code += f"        result = self.instance.{method_name}({args_str})\n"
                    if output is not None:
                        code += f"        assert result == {output}\n"
                    # No assertion for void methods (output is None, no exception raised)
                    code += "\n"
                
                for i, case in enumerate(exception_cases):
                    inputs = case['input']
                    args_str = ", ".join(str(x) for x in inputs)
                    exc_type = case.get('exception', 'Exception')
                    
                    code += f"    def test_{method_name}_exception_{i+1}(self):\n"
                    code += f"        with pytest.raises({exc_type}):\n"
                    code += f"            self.instance.{method_name}({args_str})\n\n"
            
            code += "\n"  # Empty line between test classes
    else:
        # Standalone fonksiyonlar için
        all_funcs = list(cases_by_function.keys())
        if all_funcs:
            code += f"from target_code import {', '.join(all_funcs)}\n\n"
        
        for func_name, cases in cases_by_function.items():
            # Her fonksiyon için MAX 10 normal + 5 exception test
            normal_cases = [c for c in cases if not c.get('exception')][:10]
            exception_cases = [c for c in cases if c.get('exception')][:5]
            
            total = len(normal_cases) + len(exception_cases)
            code += f"# ===== {func_name}: {total} tests (from {len(cases)} generated cases) =====\n\n"
            
            for i, case in enumerate(normal_cases):
                inputs = case['input']
                args_str = ", ".join(str(x) for x in inputs)
                output = case.get('output')
                
                code += f"def test_{func_name}_case_{i+1}():\n"
                code += f"    result = {func_name}({args_str})\n"
                if output is not None:
                    code += f"    assert result == {output}\n"
                # No assertion for void functions
                code += "\n"
            
            for i, case in enumerate(exception_cases):
                inputs = case['input']
                args_str = ", ".join(str(x) for x in inputs)
                exc_type = case.get('exception', 'Exception')
                
                code += f"def test_{func_name}_exception_{i+1}():\n"
                code += f"    with pytest.raises({exc_type}):\n"
                code += f"        {func_name}({args_str})\n\n"
    
    return code

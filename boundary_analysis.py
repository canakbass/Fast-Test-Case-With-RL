"""
Boundary Value Analysis Module
------------------------------
AST ile koddan koşulları analiz eder ve sınır değerleri çıkarır.
RL sistemine akıllı input üretimi için rehberlik eder.
"""

import ast
from typing import List, Dict, Set, Tuple, Any

class BoundaryValueExtractor(ast.NodeVisitor):
    """
    AST ziyaretçisi - koddan sınır değerlerini çıkarır.
    """
    
    def __init__(self):
        self.boundaries = []  # (variable, operator, value) listesi
        self.variables = set()  # Fonksiyon parametreleri
        self.conditions = []  # Tüm koşullar
        
    def extract_comparison_value(self, node):
        """Karşılaştırma değerini çıkar"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # Python 3.7 uyumu
            return node.n
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            val = self.extract_comparison_value(node.operand)
            if val is not None:
                return -val
        return None
    
    def extract_variable_name(self, node):
        """Değişken adını çıkar"""
        if isinstance(node, ast.Name):
            return node.id
        return None
    
    def visit_FunctionDef(self, node):
        """Fonksiyon parametrelerini topla"""
        for arg in node.args.args:
            self.variables.add(arg.arg)
        self.generic_visit(node)
    
    def visit_Compare(self, node):
        """Karşılaştırma ifadelerini analiz et"""
        left = node.left
        
        for op, comparator in zip(node.ops, node.comparators):
            var_name = self.extract_variable_name(left)
            value = self.extract_comparison_value(comparator)
            
            # Değişken sol tarafta değilse, sağ tarafı kontrol et
            if var_name is None:
                var_name = self.extract_variable_name(comparator)
                value = self.extract_comparison_value(left)
                # Operatörü tersine çevir
                op = self._reverse_op(op)
            
            if var_name and value is not None:
                op_str = self._op_to_str(op)
                self.boundaries.append({
                    'variable': var_name,
                    'operator': op_str,
                    'value': value,
                    'boundary_values': self._generate_boundary_values(value, op_str)
                })
                self.conditions.append(f"{var_name} {op_str} {value}")
        
        self.generic_visit(node)
    
    def _reverse_op(self, op):
        """Operatörü tersine çevir"""
        reverse_map = {
            ast.Lt: ast.Gt,
            ast.Gt: ast.Lt,
            ast.LtE: ast.GtE,
            ast.GtE: ast.LtE,
            ast.Eq: ast.Eq,
            ast.NotEq: ast.NotEq,
        }
        # If operator type is in map, instantiate the class. Otherwise return original op
        if type(op) in reverse_map:
            return reverse_map[type(op)]()
        else:
            return op
    
    def _op_to_str(self, op):
        """Operatörü string'e çevir"""
        op_map = {
            ast.Lt: '<',
            ast.Gt: '>',
            ast.LtE: '<=',
            ast.GtE: '>=',
            ast.Eq: '==',
            ast.NotEq: '!=',
        }
        return op_map.get(type(op), '?')
    
    def _generate_boundary_values(self, value, operator):
        """Sınır değerlerini üret"""
        if not isinstance(value, (int, float)):
            return [value]
        
        # Temel sınır değerleri: değerin kendisi, -1, +1
        boundaries = set()
        boundaries.add(value)
        boundaries.add(value - 1)
        boundaries.add(value + 1)
        
        # Operatöre göre ek değerler
        if operator in ['<', '<=']:
            boundaries.add(value - 2)
        elif operator in ['>', '>=']:
            boundaries.add(value + 2)
        elif operator == '==':
            boundaries.add(value - 1)
            boundaries.add(value + 1)
        elif operator == '!=':
            boundaries.add(value)
        
        # Sıfır etrafı her zaman önemli
        if abs(value) <= 5:
            boundaries.update([-2, -1, 0, 1, 2])
        
        return sorted(list(boundaries))


def extract_boundary_values(code_content: str) -> Dict:
    """
    Koddan sınır değerlerini çıkar.
    
    Returns:
        {
            'boundaries': [...],
            'conditions': [...],
            'critical_values': [...],
            'test_inputs': [...]
        }
    """
    try:
        tree = ast.parse(code_content)
    except SyntaxError as e:
        return {'error': str(e), 'boundaries': [], 'critical_values': []}
    
    extractor = BoundaryValueExtractor()
    extractor.visit(tree)
    
    # Tüm kritik değerleri topla
    critical_values = set()
    for b in extractor.boundaries:
        critical_values.update(b['boundary_values'])
    
    # Her zaman test edilmesi gereken değerler
    critical_values.update([-1, 0, 1, 2])
    
    # Fonksiyon parametreleri için önerilen test inputları
    test_inputs = generate_test_inputs(extractor.boundaries, extractor.variables)
    
    return {
        'boundaries': extractor.boundaries,
        'conditions': extractor.conditions,
        'variables': list(extractor.variables),
        'critical_values': sorted(list(critical_values)),
        'test_inputs': test_inputs
    }


def generate_test_inputs(boundaries: List[Dict], variables: Set[str]) -> List[Dict]:
    """
    Sınır değerlerine göre test inputları üret.
    """
    test_inputs = []
    
    # Her değişken için kritik değerleri topla
    var_values = {}
    for var in variables:
        var_values[var] = set([-1, 0, 1, 2])  # Default değerler
    
    for b in boundaries:
        var = b['variable']
        if var in var_values:
            var_values[var].update(b['boundary_values'])
    
    # Kombinasyonlar oluştur (basit versiyon)
    if len(variables) == 0:
        return []
    
    var_list = list(variables)
    
    if len(var_list) == 1:
        var = var_list[0]
        for val in sorted(var_values.get(var, [-1, 0, 1])):
            test_inputs.append({var: val})
    
    elif len(var_list) == 2:
        var1, var2 = var_list[0], var_list[1]
        vals1 = sorted(var_values.get(var1, [-1, 0, 1]))[:5]  # Max 5
        vals2 = sorted(var_values.get(var2, [-1, 0, 1]))[:5]
        for v1 in vals1:
            for v2 in vals2:
                test_inputs.append({var1: v1, var2: v2})
    
    elif len(var_list) >= 3:
        # 3+ parametre için sadece kritik kombinasyonlar
        vals = {}
        for var in var_list[:3]:
            vals[var] = sorted(var_values.get(var, [-1, 0, 1]))[:3]
        
        # Kartezyen çarpım yerine akıllı kombinasyon
        for i, var in enumerate(var_list[:3]):
            for val in vals[var]:
                input_dict = {}
                for j, v in enumerate(var_list[:3]):
                    if i == j:
                        input_dict[v] = val
                    else:
                        input_dict[v] = 0  # Default
                test_inputs.append(input_dict)
    
    return test_inputs


def analyze_code_for_testing(code_content: str) -> Dict:
    """
    Kodun tam analizini yap - sınır değerleri, fonksiyonlar, parametreler.
    """
    result = {
        'functions': [],
        'boundary_analysis': extract_boundary_values(code_content),
        'suggested_test_strategy': []
    }
    
    try:
        tree = ast.parse(code_content)
    except SyntaxError:
        return result
    
    # Fonksiyonları analiz et
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_info = {
                'name': node.name,
                'params': [],
                'has_error_handling': False,
                'branches': 0
            }
            
            # Parametreler
            for arg in node.args.args:
                param = {'name': arg.arg, 'type': None}
                if arg.annotation:
                    if isinstance(arg.annotation, ast.Name):
                        param['type'] = arg.annotation.id
                func_info['params'].append(param)
            
            # Branch sayısı
            for child in ast.walk(node):
                if isinstance(child, (ast.If, ast.While, ast.For)):
                    func_info['branches'] += 1
                if isinstance(child, (ast.Raise, ast.Try)):
                    func_info['has_error_handling'] = True
            
            result['functions'].append(func_info)
            
            # Test stratejisi öner
            strategy = f"Fonksiyon '{node.name}': "
            if func_info['has_error_handling']:
                strategy += "Hata durumlarını test et. "
            if func_info['branches'] > 2:
                strategy += f"{func_info['branches']} branch var, tüm yolları test et. "
            if len(func_info['params']) > 0:
                strategy += f"Parametreler: {[p['name'] for p in func_info['params']]}"
            
            result['suggested_test_strategy'].append(strategy)
    
    return result


# ============== RL İÇİN ENTEGRASYON ==============

def get_smart_initial_values(code_content: str, num_params: int) -> List[List[float]]:
    """
    RL ajanı için akıllı başlangıç değerleri üret.
    Bu değerler sınır değer analizine dayalı.
    """
    analysis = extract_boundary_values(code_content)
    critical = analysis.get('critical_values', [-1, 0, 1, 2])
    
    # Kritik değerlerden kombinasyonlar oluştur
    smart_values = []
    
    # Tek parametre
    for val in critical[:10]:
        smart_values.append([float(val)] * num_params)
    
    # Karışık değerler
    for i, val in enumerate(critical[:5]):
        combo = [0.0] * num_params
        combo[i % num_params] = float(val)
        smart_values.append(combo)
    
    # Hepsi aynı kritik değer
    for val in [-1, 0, 1, 2]:
        smart_values.append([float(val)] * num_params)
    
    return smart_values


def calculate_boundary_reward(action: List[float], code_content: str) -> float:
    """
    Action'ın sınır değerlere ne kadar yakın olduğuna göre bonus reward.
    """
    analysis = extract_boundary_values(code_content)
    critical = set(analysis.get('critical_values', [-1, 0, 1, 2]))
    
    bonus = 0.0
    for val in action:
        int_val = int(round(val))
        if int_val in critical:
            bonus += 2.0  # Tam sınır değer
        elif int_val - 1 in critical or int_val + 1 in critical:
            bonus += 0.5  # Yakın değer
    
    return bonus


# ============== TEST ==============

if __name__ == "__main__":
    test_code = '''
def factorial(n):
    if n < 0:
        raise ValueError("Negative number")
    if n == 0:
        return 1
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

def is_prime(n):
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
    
    print("=== Boundary Value Analysis ===\n")
    
    analysis = analyze_code_for_testing(test_code)
    
    print("Fonksiyonlar:")
    for func in analysis['functions']:
        print(f"  - {func['name']}: {len(func['params'])} param, {func['branches']} branch")
    
    print("\nKoşullar:")
    for cond in analysis['boundary_analysis']['conditions']:
        print(f"  - {cond}")
    
    print("\nSınır Değerleri:")
    for b in analysis['boundary_analysis']['boundaries']:
        print(f"  - {b['variable']} {b['operator']} {b['value']}")
        print(f"    Test değerleri: {b['boundary_values']}")
    
    print("\nKritik Değerler:")
    print(f"  {analysis['boundary_analysis']['critical_values']}")
    
    print("\nÖnerilen Test Stratejisi:")
    for strategy in analysis['suggested_test_strategy']:
        print(f"  - {strategy}")
    
    print("\nRL için Akıllı Başlangıç Değerleri:")
    smart_vals = get_smart_initial_values(test_code, 2)
    for vals in smart_vals[:5]:
        print(f"  {vals}")

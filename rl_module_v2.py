"""
RL-based Test Case Generation Module v2
---------------------------------------
Doğru mimari ile yeniden yazıldı.

Temel prensipler:
1. Sabit observation/action space (genelleştirilebilir)
2. Tek bir base model, tüm kodlarda çalışır
3. Coverage-based reward + Boundary Value Bonus
4. Transfer learning desteği
"""

import warnings
warnings.filterwarnings('ignore', message='.*Gym.*')
warnings.filterwarnings('ignore', category=DeprecationWarning)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import coverage
import inspect
import importlib.util
import os
import ast
import sys
import torch
import tempfile
import shutil
import atexit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

# Boundary Analysis entegrasyonu
from boundary_analysis import extract_boundary_values, calculate_boundary_reward, get_smart_initial_values

# ============== TEMP FOLDER YÖNETİMİ ==============

TEMP_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
_temp_files = []  # Oluşturulan temp dosyalarını takip et

def ensure_temp_folder():
    """Temp klasörünü oluştur."""
    if not os.path.exists(TEMP_FOLDER):
        os.makedirs(TEMP_FOLDER)
    return TEMP_FOLDER

def cleanup_temp_files():
    """Tüm temp dosyalarını sil."""
    for f in _temp_files:
        try:
            if os.path.exists(f):
                os.remove(f)
        except:
            pass
    _temp_files.clear()
    # Temp klasördeki tüm dosyaları da temizle
    if os.path.exists(TEMP_FOLDER):
        try:
            shutil.rmtree(TEMP_FOLDER)
        except:
            pass

# Program kapanırken temizlik yap
atexit.register(cleanup_temp_files)

# ============== KOD ANALİZ FONKSİYONLARI ==============

def extract_code_features(code_content):
    """
    Koddan sabit boyutlu feature vektörü çıkar.
    Bu, observation space için kullanılacak.
    """
    try:
        tree = ast.parse(code_content)
    except:
        return np.zeros(20, dtype=np.float32)
    
    features = {
        'num_functions': 0,
        'num_classes': 0,
        'num_if': 0,
        'num_for': 0,
        'num_while': 0,
        'num_try': 0,
        'num_return': 0,
        'num_args': 0,
        'max_depth': 0,
        'num_comparisons': 0,
        'num_bool_ops': 0,
        'num_math_ops': 0,
        'num_calls': 0,
        'num_assignments': 0,
        'num_lines': len(code_content.split('\n')),
        'has_recursion': 0,
        'num_params_first_func': 0,
        'num_branches': 0,
        'complexity_estimate': 0,
        'num_exceptions': 0,
    }
    
    function_names = set()
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            features['num_functions'] += 1
            function_names.add(node.name)
            if features['num_params_first_func'] == 0:
                features['num_params_first_func'] = len(node.args.args)
        elif isinstance(node, ast.ClassDef):
            features['num_classes'] += 1
        elif isinstance(node, ast.If):
            features['num_if'] += 1
            features['num_branches'] += 1
        elif isinstance(node, ast.For):
            features['num_for'] += 1
        elif isinstance(node, ast.While):
            features['num_while'] += 1
        elif isinstance(node, ast.Try):
            features['num_try'] += 1
        elif isinstance(node, ast.Return):
            features['num_return'] += 1
        elif isinstance(node, ast.Compare):
            features['num_comparisons'] += 1
        elif isinstance(node, ast.BoolOp):
            features['num_bool_ops'] += 1
        elif isinstance(node, ast.BinOp):
            features['num_math_ops'] += 1
        elif isinstance(node, ast.Call):
            features['num_calls'] += 1
            if isinstance(node.func, ast.Name) and node.func.id in function_names:
                features['has_recursion'] = 1
        elif isinstance(node, ast.Assign):
            features['num_assignments'] += 1
        elif isinstance(node, ast.Raise):
            features['num_exceptions'] += 1
    
    # Complexity estimate
    features['complexity_estimate'] = (
        features['num_if'] + features['num_for'] + features['num_while'] +
        features['num_try'] + features['num_comparisons']
    )
    
    # Normalize ve array'e çevir
    feature_vector = np.array(list(features.values()), dtype=np.float32)
    # Normalize (0-1 arası)
    feature_vector = feature_vector / (feature_vector.max() + 1e-8)
    
    return feature_vector


def get_function_info(code_content, module_name="temp", target_func_name=None, is_method=False, class_name=None):
    """
    Koddan fonksiyon ve parametre bilgilerini çıkar.
    Class metodları için is_method=True ve class_name gerekli.
    """
    ensure_temp_folder()
    file_path = os.path.join(TEMP_FOLDER, f"{module_name}_temp.py")
    _temp_files.append(file_path)
    
    with open(file_path, "w", encoding='utf-8') as f:
        f.write(code_content)
    
    # AST ile fonksiyon adını bul (hedef verilmemişse)
    func_name_to_find = target_func_name
    if not func_name_to_find:
        try:
            tree = ast.parse(code_content)
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name_to_find = node.name
                    break
        except SyntaxError:
            pass
    
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Class metodu ise
        if is_method and class_name:
            if hasattr(module, class_name):
                cls = getattr(module, class_name)
                # Class instance oluştur - __init__ parametrelerini kontrol et
                instance = None
                try:
                    # Önce parametresiz dene
                    instance = cls()
                except TypeError:
                    # __init__ parametreleri varsa, varsayılan değerlerle dene
                    try:
                        init_sig = inspect.signature(cls.__init__)
                        init_args = {}
                        for p in init_sig.parameters.values():
                            if p.name == 'self':
                                continue
                            # Tip bazlı varsayılan değerler
                            if p.annotation != inspect.Parameter.empty:
                                if p.annotation == int or p.annotation == 'int':
                                    init_args[p.name] = 10
                                elif p.annotation == float or p.annotation == 'float':
                                    init_args[p.name] = 10.0
                                elif p.annotation == str or p.annotation == 'str':
                                    init_args[p.name] = "test"
                                elif p.annotation == list or p.annotation == 'list':
                                    init_args[p.name] = []
                                else:
                                    init_args[p.name] = 10  # Varsayılan
                            else:
                                init_args[p.name] = 10
                        instance = cls(**init_args)
                    except:
                        instance = None
                except:
                    instance = None
                
                if hasattr(cls, func_name_to_find):
                    method = getattr(cls, func_name_to_find)
                    sig = inspect.signature(method)
                    params = []
                    for p in sig.parameters.values():
                        if p.name == 'self':
                            continue
                        param_info = {
                            'name': p.name,
                            'annotation': p.annotation if p.annotation != inspect.Parameter.empty else None,
                            'default': p.default if p.default != inspect.Parameter.empty else None
                        }
                        params.append(param_info)
                    
                    # Bound method döndür (instance ile)
                    if instance:
                        bound_method = getattr(instance, func_name_to_find)
                        return bound_method, params, file_path
                    else:
                        return method, params, file_path
        
        # Standalone fonksiyon
        if func_name_to_find and hasattr(module, func_name_to_find):
            obj = getattr(module, func_name_to_find)
            if inspect.isfunction(obj):
                sig = inspect.signature(obj)
                params = []
                for p in sig.parameters.values():
                    param_info = {
                        'name': p.name,
                        'annotation': p.annotation if p.annotation != inspect.Parameter.empty else None,
                        'default': p.default if p.default != inspect.Parameter.empty else None
                    }
                    params.append(param_info)
                return obj, params, file_path
        
        # Fallback: herhangi bir fonksiyon bul
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and obj.__module__ == module_name:
                sig = inspect.signature(obj)
                params = []
                for p in sig.parameters.values():
                    param_info = {
                        'name': p.name,
                        'annotation': p.annotation if p.annotation != inspect.Parameter.empty else None,
                        'default': p.default if p.default != inspect.Parameter.empty else None
                    }
                    params.append(param_info)
                return obj, params, file_path
    except Exception as e:
        print(f"Function extraction error: {e}")
    
    return None, [], file_path


def get_all_functions(code_content):
    """
    Koddaki tüm fonksiyonları ve class metodlarını çıkar.
    Sadece sayısal parametreli fonksiyonları döndür (list, str parametrelileri atla).
    """
    functions = []
    
    def is_numeric_compatible(annotation):
        """Annotation'ın sayısal olup olmadığını kontrol et"""
        # None annotation'ları reddet - tipi belirsiz
        if annotation is None:
            return False
        if isinstance(annotation, ast.Name):
            # int, float, bool kabul et; list, str, dict vs. reddet
            return annotation.id in ('int', 'float', 'bool', 'Union', 'Optional')
        if isinstance(annotation, ast.Subscript):
            # List[int], Dict[str, int] gibi complex tipler
            if isinstance(annotation.value, ast.Name):
                if annotation.value.id in ('List', 'Dict', 'Tuple', 'Set'):
                    return False
                if annotation.value.id in ('Union', 'Optional'):
                    return True  # Union[int, float] gibi
            return False
        if isinstance(annotation, ast.Constant):
            return True
        return True  # Bilinmeyen annotation'lar için True döndür
    
    def check_params(args):
        """Parametrelerin sayısal uyumlu olup olmadığını kontrol et"""
        for arg in args:
            # self parametresini atla
            if arg.arg == 'self':
                continue
            if arg.annotation and not is_numeric_compatible(arg.annotation):
                return False
        return True
    
    def count_params(args):
        """self hariç parametre sayısını say"""
        return len([a for a in args if a.arg != 'self'])
    
    try:
        tree = ast.parse(code_content)
        
        for node in ast.iter_child_nodes(tree):
            # Standalone fonksiyonlar
            if isinstance(node, ast.FunctionDef):
                if check_params(node.args.args):
                    functions.append({
                        'name': node.name,
                        'lineno': node.lineno,
                        'num_params': count_params(node.args.args),
                        'is_method': False,
                        'class_name': None
                    })
            
            # Class içindeki metodlar
            elif isinstance(node, ast.ClassDef):
                class_name = node.name
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        # Magic metodları filtrele ama __init__'i dahil et
                        if item.name.startswith('_') and item.name != '__init__':
                            continue
                        if check_params(item.args.args):
                            functions.append({
                                'name': item.name,
                                'lineno': item.lineno,
                                'num_params': count_params(item.args.args),
                                'is_method': True,
                                'class_name': class_name
                            })
    except SyntaxError as e:
        print(f"Syntax error parsing code: {e}")
    
    return functions


# ============== RL ENVIRONMENT ==============

class TestCaseGeneratorEnv(gym.Env):
    """
    RL Environment for Test Case Generation
    
    Observation: Kod özellikleri (sabit 20 boyut) + mevcut coverage durumu (10 boyut) + boundary info (5)
    Action: Test input değerleri (sabit 5 boyut, [-100, 100] arası)
    Reward: Yeni coverage artışı + boundary değer bonusu + exception bonusu
    """
    
    def __init__(self, max_params=5):
        super().__init__()
        
        self.max_params = max_params
        
        # Sabit boyutlu spaces
        # Observation: kod features (20) + coverage state (10) + param info (5)
        self.observation_space = spaces.Box(
            low=-1, high=1, 
            shape=(35,), 
            dtype=np.float32
        )
        
        # Action: max 5 parametre için değerler
        self.action_space = spaces.Box(
            low=-100, high=100, 
            shape=(max_params,), 
            dtype=np.float32
        )
        
        # State variables
        self.current_code = None
        self.current_func = None
        self.current_params = []
        self.current_file_path = None
        self.code_features = np.zeros(20, dtype=np.float32)
        self.coverage_state = np.zeros(10, dtype=np.float32)
        self.param_info = np.zeros(5, dtype=np.float32)
        
        self.covered_lines = set()
        self.total_lines = 0
        self.useful_cases = []
        self.step_count = 0
        self.test_count = 0  # Total test cases generated this episode
        self.efficiency_history = []  # Track coverage/test ratio
        self.max_steps = 30  # Azaltıldı: 100 -> 50 -> 30 (çok daha az brute force)
        
        # Boundary analysis için
        self.boundary_info = None
        self.critical_values = set()
        self.found_exceptions = set()
    
    def _count_executable_lines(self, code_content, func_name=None):
        """
        sys.settrace'in göreceği çalıştırılabilir satır sayısını hesapla.
        
        func_name verilirse sadece o fonksiyonu say, verilmezse tüm fonksiyonları say.
        
        Trace şunları GÖRMEZ:
        - def satırı (fonksiyon tanımı)
        - class satırı
        - Boş satırlar
        - Yorum satırları (#)
        - Docstring'ler (''' veya \""")
        - import satırları (modül seviyesinde)
        - decorator satırları (@)
        
        Trace şunları GÖRÜR:
        - Atama satırları (x = 5)
        - if/elif/else satırları
        - for/while satırları
        - return satırları
        - raise satırları
        - Fonksiyon çağrıları
        """
        try:
            tree = ast.parse(code_content)
        except SyntaxError:
            # Parse edilemezse basit sayım yap
            count = 0
            for line in code_content.split('\n'):
                stripped = line.strip()
                if stripped and not stripped.startswith('#') and not stripped.startswith('def '):
                    count += 1
            return max(count, 1)
        
        # AST ile çalıştırılabilir statement'ları say
        executable_count = 0
        
        for node in ast.walk(tree):
            # Fonksiyon içindeki çalıştırılabilir statement'lar
            if isinstance(node, ast.FunctionDef):
                # Eğer func_name verilmişse sadece o fonksiyonu say
                if func_name is not None and node.name != func_name:
                    continue
                    
                for stmt in ast.walk(node):
                    # Bunlar çalıştırılabilir statement'lar
                    if isinstance(stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign,  # Atamalar
                                        ast.Return, ast.Raise,  # Return/Raise
                                        ast.If, ast.For, ast.While,  # Kontrol akışı
                                        ast.Expr,  # Expression (fonksiyon çağrıları dahil)
                                        ast.Assert, ast.Pass, ast.Break, ast.Continue)):
                        # Docstring'i hariç tut (ilk Expr ve Constant/Str ise)
                        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, (ast.Constant, ast.Str)):
                            continue
                        executable_count += 1
        
        return max(executable_count, 1)
        
    def set_code(self, code_content, module_name="temp", target_func_name=None, is_method=False, class_name=None):
        """Yeni kod set et. Class metodları için is_method=True ve class_name gerekli."""
        self.current_code = code_content
        self.code_features = extract_code_features(code_content)
        self.current_func, self.current_params, self.current_file_path = get_function_info(
            code_content, module_name, target_func_name, is_method, class_name
        )
        
        # Boundary analysis
        self.boundary_info = extract_boundary_values(code_content)
        self.critical_values = set(self.boundary_info.get('critical_values', [-1, 0, 1, 2]))
        
        # Param info
        num_params = len(self.current_params)
        self.param_info = np.zeros(5, dtype=np.float32)
        self.param_info[0] = min(num_params / 5, 1.0)  # Normalized param count
        self.param_info[1] = min(len(self.critical_values) / 20, 1.0)  # Boundary count
        
        # Çalıştırılabilir satır sayısını hesapla
        func_name = target_func_name or (self.current_func.__name__ if self.current_func else None)
        self.total_lines = self._count_executable_lines(code_content, func_name)
        
        # Reset coverage
        self.covered_lines = set()
        self.all_covered_lines = set()
        self.coverage_state = np.zeros(10, dtype=np.float32)
        self.useful_cases = []
        self.found_exceptions = set()
        self.all_found_exceptions = set()
        
    def _get_observation(self):
        """Observation vektörünü oluştur"""
        return np.concatenate([
            self.code_features,
            self.coverage_state,
            self.param_info
        ]).astype(np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.covered_lines = set()
        self.coverage_state = np.zeros(10, dtype=np.float32)
        self.step_count = 0
        self.test_count = 0
        self.efficiency_history = []
        self.found_exceptions = set()
        # useful_cases sıfırlanmasın, biriksin
        return self._get_observation(), {}
    
    def step(self, action):
        self.step_count += 1
        
        if self.current_func is None:
            # Fonksiyon yoksa, 0 reward
            return self._get_observation(), 0, True, False, {}
        
        # Action'dan input değerleri çıkar
        num_params = len(self.current_params)
        if num_params == 0:
            inputs = []
        else:
            inputs = action[:num_params].tolist()
        
        # Parametreleri uygun tipe çevir
        converted_inputs = []
        for i, (val, param) in enumerate(zip(inputs, self.current_params)):
            annotation = param.get('annotation')
            if annotation == int:
                converted_inputs.append(int(val))
            elif annotation == float:
                converted_inputs.append(float(val))
            elif annotation == str:
                # Sayıyı string'e çevir
                converted_inputs.append(str(int(val)))
            elif annotation == list:
                # Basit liste oluştur
                converted_inputs.append([int(val)])
            elif hasattr(annotation, '__origin__'):
                # Tuple tip desteği
                if annotation.__origin__ == tuple:
                    # Tuple[int, int] gibi durumlarda tek değeri kullan
                    tuple_values = []
                    if hasattr(annotation, '__args__'):
                        for arg_type in annotation.__args__:
                            if arg_type == int:
                                tuple_values.append(int(val))
                            elif arg_type == str:
                                tuple_values.append(str(int(val)))
                            else:
                                tuple_values.append(int(val))
                        converted_inputs.append(tuple(tuple_values))
                    else:
                        converted_inputs.append((int(val),))
                else:
                    converted_inputs.append(float(val))
            else:
                # Default: float olarak bırak
                converted_inputs.append(float(val))
        
        # Fonksiyonu çalıştır ve satır takibi yap
        reward = 0
        new_lines_covered = set()
        exception_caught = None
        
        try:
            # Basit yaklaşım: trace modülü ile satır takibi
            executed_lines = set()
            func_name = self.current_func.__name__
            
            def trace_lines(frame, event, arg):
                if event == 'line':
                    # Sadece bizim fonksiyonumuzun satırlarını takip et
                    if frame.f_code.co_name == func_name:
                        executed_lines.add(frame.f_lineno)
                return trace_lines
            
            # Trace'i etkinleştir
            old_trace = sys.gettrace()
            sys.settrace(trace_lines)
            
            result = None
            try:
                result = self.current_func(*converted_inputs)
            except Exception as e:
                exception_caught = type(e).__name__
            finally:
                sys.settrace(old_trace)
            
            # Executed lines'ı coverage olarak kullan
            if executed_lines:
                new_lines_covered = executed_lines - self.covered_lines
            
            # === EFFICIENCY-BASED REWARD HESAPLAMA ===
            
            # Test sayısını artır
            self.test_count += 1
            
            # 1. Coverage reward: yeni satır sayısı (ana reward)
            coverage_reward = len(new_lines_covered) * 10  # Yeni satır bulmak çok değerli
            
            # 2. Efficiency reward: coverage/test ratio
            # Az test ile çok coverage yapanı ödüllendir
            current_coverage = len(self.covered_lines)
            efficiency = current_coverage / (self.test_count + 1)
            efficiency_reward = efficiency * 5
            
            # 3. Duplicate penalty: yeni coverage yoksa cezalandır
            duplicate_penalty = 0
            if len(new_lines_covered) == 0:
                duplicate_penalty = -2  # Tekrar eden test case cezası
            
            # 4. Boundary bonus: sınır değerleri keşfetme
            boundary_bonus = calculate_boundary_reward(converted_inputs, self.current_code)
            
            # 5. Exception bonus: yeni exception keşfetme
            exception_bonus = 0
            if exception_caught and exception_caught not in self.found_exceptions:
                self.found_exceptions.add(exception_caught)
                self.all_found_exceptions.add(exception_caught)  # Kümülatif
                exception_bonus = 15  # Yeni exception bulma bonusu (artırıldı)
            
            # 6. Kritik değer bonusu: 0, 1, -1 gibi değerler
            critical_bonus = 0
            for val in converted_inputs:
                int_val = int(round(val))
                if int_val in self.critical_values:
                    critical_bonus += 0.5
            
            # Toplam reward: efficiency odaklı
            reward = (coverage_reward + efficiency_reward + boundary_bonus + 
                     exception_bonus + critical_bonus + duplicate_penalty)
            
            # Efficiency'yi kaydet
            if current_coverage > 0:
                self.efficiency_history.append(efficiency)
            
            # Her çalıştırılan case'i kaydet (test generation için)
            # Sadece başarılı çalışanları veya exception fırlatanları kaydet
            self.useful_cases.append({
                'input': converted_inputs,
                'output': result,
                'new_lines': list(new_lines_covered),
                'coverage_increase': len(new_lines_covered),
                'exception': exception_caught,
                'boundary_hit': boundary_bonus > 0,
                'total_reward': reward
            })
            
            self.covered_lines.update(new_lines_covered)
            self.all_covered_lines.update(new_lines_covered)  # Kümülatif
            
            # Coverage state güncelle
            coverage_pct = len(self.covered_lines) / max(self.total_lines, 1)
            self.coverage_state[0] = coverage_pct
            self.coverage_state[1] = len(self.covered_lines) / 100  # Normalized
            self.coverage_state[2] = reward / 10  # Son reward
            self.coverage_state[3] = len(self.found_exceptions) / 5  # Exception count
                
        except Exception as e:
            pass  # Trace hatası - sessizce geç
        
        # Episode bitti mi?
        terminated = False
        truncated = self.step_count >= self.max_steps
        
        # Early stopping conditions (efficiency-based)
        coverage_pct = len(self.covered_lines) / max(self.total_lines, 1)
        
        # 1. Coverage %85'e ulaştıysa bitir (daha agresif)
        if coverage_pct >= 0.85:
            terminated = True
        
        # 2. Son 10 testte yeni coverage yoksa bitir (kompleks fonksiyonlar için)
        if self.test_count >= 10:
            recent_cases = self.useful_cases[-10:]
            if all(c.get('coverage_increase', 0) == 0 for c in recent_cases):
                terminated = True
        
        return self._get_observation(), reward, terminated, truncated, {}


# ============== CALLBACK FOR PROGRESS ==============

class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None
        
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training", unit="step")
        
    def _on_step(self):
        self.pbar.update(1)
        return True
    
    def _on_training_end(self):
        self.pbar.close()


# ============== MODEL KAYDETME/YÜKLEME ==============

def save_model(model, path="ppo_testgen_model.zip"):
    """Model kaydet"""
    model.save(path)
    print(f"Model kaydedildi: {path}")

def load_model(path="ppo_testgen_model.zip", env=None):
    """Model yükle"""
    if env is None:
        env = TestCaseGeneratorEnv()
    return PPO.load(path, env=env)


# ============== BASE MODEL EĞİTİMİ ==============

def train_base_model(
    dataset, 
    timesteps_per_code=50000,
    model_path="ppo_testgen_base.zip",
    checkpoint_freq=10000
):
    """
    Base RL modelini eğit.
    
    dataset: List of (code_content, module_name)
    timesteps_per_code: Her kod için kaç timestep
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Dataset size: {len(dataset)} kod")
    print(f"Timesteps per code: {timesteps_per_code}")
    print(f"Total timesteps: {len(dataset) * timesteps_per_code}")
    
    env = TestCaseGeneratorEnv()
    
    # Mevcut model varsa yükle, yoksa yeni oluştur
    if os.path.exists(model_path):
        print(f"Mevcut model yükleniyor: {model_path}")
        model = PPO.load(model_path, env=env, device=device)
    else:
        print("Yeni model oluşturuluyor...")
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=0, 
            device=device,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )
    
    total_codes = len(dataset)
    
    for idx, (code_content, module_name) in enumerate(dataset):
        print(f"\n[{idx+1}/{total_codes}] Training on: {module_name}")
        
        # Kodu environment'a set et
        env.set_code(code_content, module_name)
        model.set_env(env)
        
        # Eğit
        try:
            callback = ProgressCallback(timesteps_per_code)
            model.learn(
                total_timesteps=timesteps_per_code,
                reset_num_timesteps=False,
                callback=callback
            )
            
            # Her kod sonrası kaydet
            save_model(model, model_path)
            
            # İstatistikler (kümülatif coverage kullan)
            print(f"  Useful cases: {len(env.useful_cases)}")
            print(f"  Coverage: {len(env.all_covered_lines)}/{env.total_lines} lines")
            if env.all_covered_lines:
                print(f"  Covered line numbers: {sorted(env.all_covered_lines)[:20]}...")
            if env.all_found_exceptions:
                print(f"  Exceptions found: {env.all_found_exceptions}")
            
        except KeyboardInterrupt:
            print("\nEğitim kesintiye uğradı! Model kaydediliyor...")
            save_model(model, model_path)
            raise
    
    print(f"\n✓ Base model eğitimi tamamlandı: {model_path}")
    return model


# ============== TEST CASE ÜRET ==============

def generate_test_cases(
    code_content, 
    model_path="ppo_testgen_base.zip",
    num_episodes=2,  # Azaltıldı: 10 -> 2 (efficiency için)
    fine_tune_steps=0,
    use_boundary_analysis=True,
    save_finetuned_model=False,
    progress_callback=None
):
    """
    Verilen koddaki TÜM fonksiyonlar için test case'ler üret.
    
    code_content: Test edilecek Python kodu (birden fazla fonksiyon içerebilir)
    model_path: Eğitilmiş model yolu
    num_episodes: Her fonksiyon için kaç episode çalıştırılacak
    fine_tune_steps: İsteğe bağlı fine-tuning adımı (her fonksiyon için)
    use_boundary_analysis: Sınır değer analizi kullan
    save_finetuned_model: Fine-tuned modeli kaydet
    progress_callback: İlerleme mesajları için callback fonksiyonu
    """
    
    # Tüm fonksiyonları bul
    all_functions = get_all_functions(code_content)
    
    if not all_functions:
        print("Hiç test edilebilir fonksiyon bulunamadı!")
        return [], {}
    
    print(f"Bulunan test edilebilir fonksiyon sayısı: {len(all_functions)}")
    for f in all_functions:
        print(f"  - {f['name']} ({f['num_params']} parametre)")
    
    if not os.path.exists(model_path):
        print(f"Model bulunamadı: {model_path}")
        print("Önce train_base_model ile model eğitin.")
        return [], {}
    
    env = TestCaseGeneratorEnv()
    model = None
    initial_model_path = model_path  # Orijinal model yolunu sakla
    
    all_cases = []
    total_coverage_info = {
        'functions': [],
        'total_cases': 0,
        'total_coverage_lines': 0,
        'total_lines': 0,
        'all_covered_lines': set(),  # Tüm fonksiyonlardan kümülatif coverage
        'file_lines': 0,  # Dosyadaki gerçek satır sayısı
        'func_total_lines': 0  # Fonksiyonların toplam satır sayısı
    }
    
    # Dosyadaki toplam satır sayısını bir kere hesapla (boş satır ve yorumları çıkar)
    code_lines = []
    for line in code_content.split('\n'):
        stripped = line.strip()
        # Boş satırları ve sadece yorum olan satırları atla
        if stripped and not stripped.startswith('#') and not stripped.startswith('"""') and not stripped.startswith("'''"):
            code_lines.append(line)
    total_coverage_info['file_lines'] = len(code_lines)
    
    print(f"Dosyadaki kod satırı sayısı: {len(code_lines)}")
    
    # Her fonksiyon için test case üret
    for idx, func_info in enumerate(all_functions):
        func_name = func_info['name']
        is_method = func_info.get('is_method', False)
        class_name = func_info.get('class_name', None)
        
        print(f"\n{'='*50}")
        if is_method:
            print(f"Metod {idx+1}/{len(all_functions)}: {class_name}.{func_name}")
        else:
            print(f"Fonksiyon {idx+1}/{len(all_functions)}: {func_name}")
        print(f"{'='*50}")
        
        if progress_callback:
            progress_callback(f"🔄 İşleniyor: {func_name} ({idx+1}/{len(all_functions)})")
        
        # Environment'ı bu fonksiyon için ayarla
        env.set_code(code_content, f"target_{func_name}", func_name, is_method, class_name)
        
        if env.current_func is None:
            print(f"  ⚠ {func_name} fonksiyonu yüklenemedi, atlanıyor...")
            continue
        
        print(f"  Parametreler: {[p['name'] for p in env.current_params]}")
        print(f"  Hedef satır sayısı: {env.total_lines}")
        
        # Model yükle - HER ZAMAN orijinal modelden başla (fine-tuning biriktirmesin)
        if fine_tune_steps > 0:
            # Fine-tune modunda her fonksiyon için base modelden başla
            model = load_model(initial_model_path, env)
        elif model is None:
            model = load_model(model_path, env)
        else:
            model.set_env(env)
        
        # Fine-tuning (her fonksiyon için - ama sadece parametreli fonksiyonlarda)
        num_params = len(env.current_params)
        if fine_tune_steps > 0 and num_params > 0:
            print(f"  🎯 Fine-tuning: {fine_tune_steps} steps...")
            if progress_callback:
                progress_callback(f"🎓 {func_name} için model eğitiliyor... ({fine_tune_steps} steps)")
            model.learn(total_timesteps=fine_tune_steps, reset_num_timesteps=False)
            print(f"  ✅ Fine-tuning tamamlandı!")
        elif num_params == 0:
            print(f"  ⏭️  Parametresiz fonksiyon, fine-tuning atlandı")
        
        func_cases = []
        
        # Parametresiz fonksiyonlar için özel işlem
        if num_params == 0:
            print(f"  ℹ️  Parametresiz fonksiyon, tek test yeterli")
            # Sadece bir kere çalıştır
            env.useful_cases = []
            obs, _ = env.reset()
            action = np.zeros(5, dtype=np.float32)  # Dummy action
            obs, reward, _, _, _ = env.step(action)
            func_cases.extend(env.useful_cases)
            print(f"  Basit test: {len(env.useful_cases)} case")
        else:
            # Parametreli fonksiyonlar için normal işlem
            # Boundary analysis ile akıllı başlangıç
            if use_boundary_analysis:
                boundary_info = extract_boundary_values(code_content)
                smart_starts = get_smart_initial_values(code_content, max(num_params, 1))
                
                env.useful_cases = []
                for smart_input in smart_starts[:5]:
                    obs, _ = env.reset()
                    action = np.array(smart_input[:5] + [0] * (5 - len(smart_input)), dtype=np.float32)
                    obs, reward, _, _, _ = env.step(action)
                
                func_cases.extend(env.useful_cases)
                print(f"  Boundary analysis: {len(env.useful_cases)} case")
            
            # Model ile test case üret
            env.useful_cases = []
            for ep in range(num_episodes):
                obs, _ = env.reset()
                done = False
                
                while not done:
                    action, _ = model.predict(obs, deterministic=False)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
            
            func_cases.extend(env.useful_cases)
            print(f"  Model: {len(env.useful_cases)} case")
        
        # Her case'e fonksiyon adı ekle
        for case in func_cases:
            case['function'] = func_name
        
        all_cases.extend(func_cases)
        
        # Coverage bilgisi
        coverage_pct = 100 * len(env.all_covered_lines) / max(env.total_lines, 1)
        print(f"  Coverage: {len(env.all_covered_lines)}/{env.total_lines} ({coverage_pct:.1f}%)")
        
        total_coverage_info['functions'].append({
            'name': func_name,
            'cases': len(func_cases),
            'coverage_lines': len(env.all_covered_lines),
            'total_lines': env.total_lines,
            'coverage_pct': coverage_pct,
            'exceptions': list(env.all_found_exceptions)
        })
        # Kümülatif coverage (overlap'leri otomatik halleder)
        total_coverage_info['all_covered_lines'].update(env.all_covered_lines)
        # Fonksiyonların toplam satır sayısını topla
        total_coverage_info['func_total_lines'] += env.total_lines
    
    # Duplicate'leri kaldır (aynı fonksiyon + aynı input)
    unique_cases = []
    seen = set()
    for case in all_cases:
        key = (case.get('function', ''), tuple(case['input']))
        if key not in seen:
            seen.add(key)
            unique_cases.append(case)
    
    total_coverage_info['total_cases'] = len(unique_cases)
    
    # Fine-tuned modeli kaydet
    if fine_tune_steps > 0 and save_finetuned_model and model is not None:
        finetuned_path = "ppo_testgen_finetuned.zip"
        model.save(finetuned_path)
        print(f"\n💾 Fine-tuned model kaydedildi: {finetuned_path}")
        if progress_callback:
            progress_callback(f"💾 Fine-tuned model kaydedildi: {finetuned_path}")
    
    # Özet
    print(f"\n{'='*50}")
    print(f"ÖZET")
    print(f"{'='*50}")
    print(f"Toplam fonksiyon: {len(all_functions)}")
    print(f"Toplam unique test case: {len(unique_cases)}")
    
    # Doğru coverage hesaplama: Fonksiyonların toplam satır sayısı üzerinden
    total_unique_covered = len(total_coverage_info['all_covered_lines'])
    func_total_lines = total_coverage_info['func_total_lines']  # Fonksiyonların toplam satırı
    overall_coverage = 100 * total_unique_covered / max(func_total_lines, 1)
    print(f"Toplam coverage: {total_unique_covered}/{func_total_lines} ({overall_coverage:.1f}%)")
    
    # Exception'ları göster
    all_exceptions = set()
    for case in unique_cases:
        if case.get('exception'):
            all_exceptions.add(case['exception'])
    if all_exceptions:
        print(f"Bulunan exception'lar: {all_exceptions}")
    
    # Sonuç bilgileri
    result_info = {
        'total_cases': len(unique_cases),
        'coverage_lines': total_unique_covered,
        'total_lines': func_total_lines,
        'coverage_pct': overall_coverage,
        'exceptions_found': list(all_exceptions),
        'functions': total_coverage_info['functions'],
        'boundary_conditions': extract_boundary_values(code_content).get('conditions', []),
        'critical_values_used': list(env.critical_values) if env else []
    }
    
    return unique_cases, result_info


# ============== TEST ==============

if __name__ == "__main__":
    # Test kodu - sınır değerleri içeren
    test_code = """
def calculate(a: int, b: int, op: int):
    if op == 1:
        return a + b
    elif op == 2:
        return a - b
    elif op == 3:
        return a * b
    elif op == 4:
        if b != 0:
            return a / b
        raise ValueError("Division by zero")
    elif op < 0:
        raise ValueError("Invalid operation")
    else:
        return -1
"""
    
    # Mini dataset ile test
    dataset = [(test_code, "calculate")]
    
    # Eğit
    print("=== Training ===")
    train_base_model(dataset, timesteps_per_code=10000, model_path="test_model.zip")
    
    # Test case üret
    print("\n=== Generating Test Cases with Boundary Analysis ===")
    cases, info = generate_test_cases(
        test_code, 
        model_path="test_model.zip", 
        num_episodes=5,
        use_boundary_analysis=True
    )
    
    print("\n=== Generated Cases ===")
    for i, case in enumerate(cases[:10]):  # İlk 10
        exc = f" [Exception: {case.get('exception')}]" if case.get('exception') else ""
        bnd = " [Boundary Hit]" if case.get('boundary_hit') else ""
        print(f"{i+1}. Input: {case['input']}, New lines: {len(case['new_lines'])}{exc}{bnd}")
    
    print("\n=== Analysis Info ===")
    print(f"Total cases: {info['total_cases']}")
    print(f"Coverage: {info['coverage_pct']:.1f}%")
    print(f"Exceptions found: {info['exceptions_found']}")
    print(f"Boundary conditions: {info['boundary_conditions']}")

# 🤖 RL Test Case Generator - Model Dokümantasyonu

## 📋 Genel Bakış

Bu sistem, **Reinforcement Learning (Pekiştirmeli Öğrenme)** kullanarak Python kodları için otomatik test case'ler üreten bir sistemdir.

---

## 🧠 Model Bilgileri

| Özellik | Değer |
|---------|-------|
| **Algoritma** | PPO (Proximal Policy Optimization) |
| **Kütüphane** | Stable-Baselines3 |
| **Policy Network** | MlpPolicy (Multi-Layer Perceptron) |
| **Framework** | PyTorch |

### PPO Neden Seçildi?
- ✅ Continuous action space desteği (test değerleri sürekli değerler)
- ✅ Stabil eğitim (clipping ile büyük güncellemeleri önler)
- ✅ Sample efficient (az veriyle iyi öğrenir)
- ✅ On-policy algoritma (güvenli ve tahmin edilebilir)

---

## 📥 Observation Space (Model Girdisi)

Model, **35 boyutlu** bir observation vektörü alır:

```
Observation Vector [35 dim] = Code Features [20] + Coverage State [10] + Param Info [5]
```

### 1. Code Features (20 boyut)
Kodun statik analizi (AST'den çıkarılır):

| Index | Feature | Açıklama |
|-------|---------|----------|
| 0 | num_functions | Fonksiyon sayısı |
| 1 | num_classes | Sınıf sayısı |
| 2 | num_if | If statement sayısı |
| 3 | num_for | For loop sayısı |
| 4 | num_while | While loop sayısı |
| 5 | num_try | Try-except sayısı |
| 6 | num_return | Return statement sayısı |
| 7 | num_args | Toplam argüman sayısı |
| 8 | max_depth | Maksimum iç içe derinlik |
| 9 | num_comparisons | Karşılaştırma operatörü sayısı (>, <, ==) |
| 10 | num_bool_ops | Boolean operatör sayısı (and, or, not) |
| 11 | num_math_ops | Matematiksel operatör sayısı (+, -, *, /) |
| 12 | num_calls | Fonksiyon çağrısı sayısı |
| 13 | num_assignments | Atama sayısı |
| 14 | num_lines | Toplam satır sayısı |
| 15 | has_recursion | Özyineleme var mı (0/1) |
| 16 | num_params_first_func | İlk fonksiyonun parametre sayısı |
| 17 | num_branches | Dal sayısı (if/elif/else) |
| 18 | complexity_estimate | Tahmini karmaşıklık skoru |
| 19 | num_exceptions | Raise/exception sayısı |

### 2. Coverage State (10 boyut)
Mevcut test coverage durumu:

| Index | Feature | Açıklama |
|-------|---------|----------|
| 0 | coverage_pct | Şu anki coverage yüzdesi (0-1) |
| 1 | new_lines_ratio | Son adımda eklenen satır oranı |
| 2 | unique_cases_ratio | Unique case oranı |
| 3 | exception_found | Exception bulundu mu (0/1) |
| 4-9 | recent_rewards | Son 6 adımın reward'ları |

### 3. Parameter Info (5 boyut)
Test edilecek fonksiyonun parametre bilgileri:

| Index | Feature | Açıklama |
|-------|---------|----------|
| 0 | num_params | Parametre sayısı (normalized) |
| 1 | has_int_param | int parametresi var mı |
| 2 | has_float_param | float parametresi var mı |
| 3 | has_str_param | str parametresi var mı |
| 4 | has_default | Default değer var mı |

---

## 📤 Action Space (Model Çıktısı)

Model, **5 boyutlu continuous** bir action vektörü üretir:

```
Action Vector [5 dim] = [param1, param2, param3, param4, param5]
```

| Özellik | Değer |
|---------|-------|
| **Tip** | Continuous (Box) |
| **Range** | [-100, +100] |
| **Boyut** | 5 (maksimum 5 parametreye kadar destekler) |

### Action → Test Input Dönüşümü

```python
# Model çıktısı (örnek)
action = [25.7, -50.3, 0.0, 0.0, 0.0]

# Fonksiyon: def calculator(a: int, b: int, op: str)
# Dönüşüm:
a = int(25.7)      # → 25
b = int(-50.3)     # → -50
op = "+" veya "-"  # (action[2] > 0 ise "+", değilse "-")
```

### Tip Dönüşüm Kuralları

| Parametre Tipi | Dönüşüm |
|----------------|---------|
| `int` | `int(action_value)` |
| `float` | `float(action_value)` |
| `str` | Predefined list'ten seçim (action sign'a göre) |
| `bool` | `action_value > 0` |
| `Tuple[int, int]` | `(int(action_value), int(action_value + 1))` |

---

## 🎯 Reward Function (Ödül Fonksiyonu)

Model şu reward'ları alır:

```python
total_reward = (
    coverage_reward * 10 +      # Yeni satır kapama
    efficiency_reward * 5 +      # Verimlilik bonusu
    exception_bonus * 15 -       # Exception bulma bonusu
    duplicate_penalty * 2        # Duplicate cezası
)
```

### Reward Bileşenleri

| Bileşen | Formül | Açıklama |
|---------|--------|----------|
| **Coverage Reward** | `new_lines_covered * 10` | Her yeni kaplanan satır için +10 |
| **Efficiency Reward** | `(new_lines / total_cases) * 5` | Az test ile çok coverage = bonus |
| **Exception Bonus** | `+15` | Yeni exception bulunursa |
| **Duplicate Penalty** | `-2` | Aynı input tekrarlanırsa |
| **Boundary Bonus** | `+3` | Sınır değer test edilirse (0, -1, MAX) |

### Early Stopping Koşulları

| Koşul | Açıklama |
|-------|----------|
| Coverage ≥ 85% | Yeterli coverage'a ulaşıldı |
| 10 ardışık duplicate | Model yeni şey üretemiyor |
| max_steps = 30 | Maksimum adım sayısına ulaşıldı |

---

## 🔄 Çalışma Senaryosu

### Senaryo: `calculator(a, b, op)` Fonksiyonu İçin Test Üretimi

```python
def calculator(a: int, b: int, op: str):
    if op == "+":
        return a + b
    elif op == "-":
        return a - b
    elif op == "*":
        return a * b
    elif op == "/":
        if b == 0:
            raise ValueError("Division by zero")
        return a / b
    else:
        raise ValueError("Unknown operator")
```

### Adım Adım İşleyiş:

```
┌─────────────────────────────────────────────────────────────────┐
│ ADIM 1: Kod Analizi                                             │
├─────────────────────────────────────────────────────────────────┤
│ • AST parse edilir                                              │
│ • Code features çıkarılır (20 dim)                              │
│ • Fonksiyon parametreleri belirlenir: [a:int, b:int, op:str]   │
│ • Toplam satır sayısı: 11                                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ ADIM 2: Environment Reset                                       │
├─────────────────────────────────────────────────────────────────┤
│ • Coverage tracker başlatılır                                   │
│ • Observation vektörü oluşturulur [35 dim]                      │
│ • covered_lines = {} (boş set)                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ ADIM 3: Model Prediction (Episode Loop)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Episode 1, Step 1:                                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Observation: [0.2, 0.1, 0.3, ..., 0.0, 0.0]             │   │
│  │              ↓                                           │   │
│  │         ┌─────────┐                                      │   │
│  │         │   PPO   │                                      │   │
│  │         │  Model  │                                      │   │
│  │         └─────────┘                                      │   │
│  │              ↓                                           │   │
│  │ Action: [10.5, 5.2, 0.8, 0.0, 0.0]                      │   │
│  │              ↓                                           │   │
│  │ Test Input: calculator(10, 5, "+")                       │   │
│  │              ↓                                           │   │
│  │ Execution: Çalıştır + Coverage ölç                       │   │
│  │              ↓                                           │   │
│  │ Result: return 15, lines covered: {1,2,3}               │   │
│  │              ↓                                           │   │
│  │ Reward: 3 new lines × 10 = +30                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Episode 1, Step 2:                                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Observation: [0.2, 0.1, 0.3, ..., 0.27, 0.3]            │   │
│  │ Action: [-15.2, 8.1, -0.5, 0.0, 0.0]                    │   │
│  │ Test Input: calculator(-15, 8, "-")                      │   │
│  │ Result: return -23, lines covered: {4,5}                │   │
│  │ Reward: 2 new lines × 10 = +20                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ... (devam eder ta ki early stopping veya max_steps)           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ ADIM 4: Sonuç                                                   │
├─────────────────────────────────────────────────────────────────┤
│ Üretilen Test Case'ler:                                         │
│ • calculator(10, 5, "+")    → 15                               │
│ • calculator(-15, 8, "-")   → -23                              │
│ • calculator(3, 4, "*")     → 12                               │
│ • calculator(10, 2, "/")    → 5.0                              │
│ • calculator(5, 0, "/")     → ValueError (Division by zero)    │
│ • calculator(1, 1, "%")     → ValueError (Unknown operator)    │
│                                                                 │
│ Coverage: 11/11 (100%)                                          │
│ Total Test Cases: 6                                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Sistem Mimarisi

```
┌──────────────────────────────────────────────────────────────────────┐
│                         FAST TEST CASE GENERATOR                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐            │
│  │   main.py   │────▶│ rl_module   │────▶│  coverage   │            │
│  │ (Streamlit) │     │   _v2.py    │     │   module    │            │
│  └─────────────┘     └─────────────┘     └─────────────┘            │
│         │                   │                   │                    │
│         │                   ▼                   │                    │
│         │           ┌─────────────┐             │                    │
│         │           │    PPO      │             │                    │
│         │           │   Model     │             │                    │
│         │           │ (SB3)       │             │                    │
│         │           └─────────────┘             │                    │
│         │                   │                   │                    │
│         ▼                   ▼                   ▼                    │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐            │
│  │  boundary   │     │   analiz    │     │    llm      │            │
│  │ _analysis   │     │  _module    │     │  _module    │            │
│  │    .py      │     │    .py      │     │    .py      │            │
│  └─────────────┘     └─────────────┘     └─────────────┘            │
│         │                   │                   │                    │
│         └───────────────────┴───────────────────┘                    │
│                             │                                        │
│                             ▼                                        │
│                    ┌─────────────────┐                               │
│                    │  pytest code    │                               │
│                    │   generation    │                               │
│                    └─────────────────┘                               │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Model Dosyaları

| Dosya | Açıklama |
|-------|----------|
| `ppo_testgen_base.zip` | Ana eğitilmiş model (10 fonksiyon, 30k step/fonksiyon) |
| `ppo_testgen_finetuned.zip` | Fine-tune edilmiş model (kullanıcı kodu üzerinde) |

### Model Yükleme

```python
from stable_baselines3 import PPO

# Model yükle
model = PPO.load("ppo_testgen_base.zip", env=env)

# Prediction yap
action, _ = model.predict(observation, deterministic=False)
```

---

## 🔧 Eğitim Parametreleri

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `learning_rate` | 3e-4 | Öğrenme oranı |
| `n_steps` | 2048 | Her update için adım sayısı |
| `batch_size` | 64 | Mini-batch boyutu |
| `n_epochs` | 10 | Her update için epoch sayısı |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_range` | 0.2 | PPO clip range |
| `total_timesteps` | 300,000 | Toplam eğitim adımı (10 fonksiyon × 30k) |

---

## 📊 Performans Metrikleri

### Base Model (10 Fonksiyon Üzerinde Eğitildi)

| Metrik | Değer |
|--------|-------|
| Ortalama Coverage | ~85% |
| Ortalama Test Case Sayısı | ~35-40 per fonksiyon |
| Exception Discovery Rate | ~90% |
| Training Time | ~15 dakika |

### Fine-tuning Sonrası

| Metrik | İyileşme |
|--------|----------|
| Coverage | +5-10% |
| Test Efficiency | +20% (daha az test, aynı coverage) |

---

## ⚠️ Bilinen Limitasyonlar

1. **Maksimum 5 Parametre**: Action space 5 boyutlu
2. **Sadece Basit Tipler**: int, float, str, bool, Tuple desteklenir
3. **Liste/Dict Desteği Yok**: Kompleks veri yapıları desteklenmiyor
4. **İç İçe Bloklar**: Derin nested if'lerde coverage düşebilir
5. **String Parametreler**: Sınırlı string seti kullanılıyor

---

## 🚀 Gelecek İyileştirmeler (Planlanan)

1. **Block-Level Targeting**: Her if bloğu için ayrı hedefleme
2. **Path Coverage**: Tüm execution path'leri kaplama
3. **Symbolic Execution**: Constraint-based test generation
4. **Daha Büyük Action Space**: 10+ parametre desteği
5. **Complex Types**: List, Dict, Custom class desteği

---

## 📚 Referanslar

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PPO Paper (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Radon Code Metrics](https://radon.readthedocs.io/)

---

*Son Güncelleme: Aralık 2024*

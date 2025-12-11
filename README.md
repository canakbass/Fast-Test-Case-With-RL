# 🧪 RL-based Automatic Test Case Generator

**Reinforcement Learning ile Otomatik Test Case Üretimi**

Bu proje, Python kodları için otomatik test case üretimi yapan bir RL (Reinforcement Learning) sistemidir. Boundary Value Analysis ve Coverage Maximization tekniklerini birleştirir.

## 🚀 Özellikler

- **RL Tabanlı Test Üretimi**: PPO algoritması ile coverage'ı maksimize eden test inputları keşfeder
- **Boundary Value Analysis**: AST ile koddan koşulları analiz eder ve sınır değerlerini belirler
- **Transfer Learning**: Tek bir base model, farklı kodlarda kullanılabilir
- **Exception Discovery**: Hata durumlarını otomatik tespit eder
- **Pytest Kod Üretimi**: LLM veya mock generator ile pytest koduna çevirir
- **Streamlit Arayüzü**: Kullanıcı dostu web arayüzü

## 📁 Proje Yapısı

```
├── rl_module_v2.py       # Ana RL modülü (PPO, Environment, Training)
├── boundary_analysis.py   # AST ile sınır değer analizi
├── llm_module.py          # LLM ile pytest kod üretimi
├── analiz_module.py       # Radon ile kod metrikleri
├── main.py                # Streamlit web arayüzü
├── train_v2.py            # Base model eğitim scripti
├── target_code.py         # Örnek test hedef kodu
├── test_integration.py    # Integration testleri
└── requirements.txt       # Bağımlılıklar
```

## 🛠️ Kurulum

```bash
# Virtual environment oluştur
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Bağımlılıkları yükle
pip install -r requirements.txt
```

## 📖 Kullanım

### 1. Base Model Eğitimi

Önce çeşitli fonksiyonlar üzerinde base modeli eğitin:

```bash
python train_v2.py
```

Bu işlem ~8-10 dakika sürer ve `ppo_testgen_base.zip` dosyasını oluşturur.

### 2. Web Arayüzü

Streamlit arayüzünü başlatın:

```bash
streamlit run main.py
```

### 3. Programatik Kullanım

```python
from rl_module_v2 import generate_test_cases

code = '''
def calculate(a: int, b: int):
    if a < 0:
        return -1
    return a + b
'''

cases, info = generate_test_cases(
    code,
    model_path="ppo_testgen_base.zip",
    num_episodes=10,
    use_boundary_analysis=True
)

print(f"Üretilen test case sayısı: {len(cases)}")
print(f"Coverage: {info['coverage_pct']:.1f}%")
```

## 🧠 Nasıl Çalışır?

### 1. Observation Space (35 boyut)
- **Kod Özellikleri (20)**: if sayısı, döngü sayısı, karmaşıklık vs.
- **Coverage State (10)**: Mevcut coverage durumu
- **Param Info (5)**: Parametre bilgileri

### 2. Action Space (5 boyut)
- Max 5 parametre için [-100, 100] arası değerler

### 3. Reward Fonksiyonu
- **Coverage Reward**: Yeni satır keşfi (+2 per satır)
- **Boundary Bonus**: Sınır değerleri vuruşu
- **Exception Bonus**: Yeni exception keşfi (+10)
- **Critical Value Bonus**: 0, 1, -1 gibi kritik değerler

### 4. Training Pipeline
1. Kod → AST analizi → Feature extraction
2. Boundary value extraction
3. RL agent training (PPO)
4. Test case generation
5. (Opsiyonel) LLM ile pytest koduna çevirme

## 📊 Örnek Çıktı

```
[Training] add: 1/2 lines covered
[Training] max_of_three: 5/7 lines covered
[Training] factorial: 8/9 lines covered
[Training] is_prime: 9/13 lines covered

Generated Test Cases:
1. Input: [5, 3]     - Normal case
2. Input: [-1, 0]    - Boundary hit
3. Input: [0, 0]     - Zero case [Exception: ValueError]
```

## 🔧 Konfigürasyon

`main.py` sidebar'dan ayarlanabilir:
- **Fine-Tune Steps**: Yeni kod için ek eğitim
- **Episode Sayısı**: Test üretim iterasyonu
- **Boundary Analysis**: Sınır değer analizi açık/kapalı

## 📝 Gereksinimler

- Python 3.9+
- PyTorch
- Stable-Baselines3
- Gymnasium
- Streamlit
- OpenAI API (opsiyonel, pytest üretimi için)

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing`)
5. Pull Request açın

## 📄 Lisans

MIT License

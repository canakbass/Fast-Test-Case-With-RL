from rl_module import train_base_model
from tqdm import tqdm

# Dataset
files = ['sample_code.py', 'target_code.py', 'temp_module.py']
dataset = [(open(f).read(), f.split('.')[0]) for f in files]

total = len(dataset)
print(f"Base RL Model Eğitimi Başlıyor: {total} kod, 2M timesteps")

for i, (code, name) in enumerate(tqdm(dataset, desc="Kodlar", unit="kod")):
    print(f"Eğitim: {name} ({i+1}/{total})")
    train_base_model([(code, name)], timesteps=200000, model_path=f'ppo_base_model_{name}.zip')
    print(f"Model kaydedildi: ppo_base_model_{name}.zip\n")

print("Tüm base modeller eğitildi!")
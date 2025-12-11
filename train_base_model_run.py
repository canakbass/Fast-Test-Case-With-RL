from rl_module import train_base_model
from dataset_large import dataset

print(f"Base RL Model Eğitimi Başlıyor: {len(dataset)} farklı fonksiyon, 2M timesteps")
print("GPU kullanılıyorsa çok daha hızlı olacak...")
print("Checkpoint sistemi aktif - kesinti durumunda devam edilebilir.\n")

# Tek bir unified base model eğit
train_base_model(dataset, timesteps=200000, model_path='ppo_base_model.zip', checkpoint_freq=50000)

print("\n✓ Base model eğitimi tamamlandı: ppo_base_model.zip")
print("\nŞimdi arayüzü başlatmak için:")
print("streamlit run main.py")

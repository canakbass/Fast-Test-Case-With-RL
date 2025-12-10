# AI Test Generator

Bu proje, Reinforcement Learning (RL) ve Large Language Model (LLM) teknolojilerini kullanarak Python fonksiyonları için otomatik olarak test case'leri ve pytest kodları üreten bir araçtır.

## Özellikler

*   **Otomatik Test Case Üretimi:** RL (PPO) ajanı, hedef fonksiyonun kod kapsamını (coverage) maksimize edecek girdileri keşfeder.
*   **Dinamik Coverage Analizi:** Çalışma zamanında fonksiyonu çalıştırarak gerçek coverage değerlerini ölçer.
*   **LLM ile Pytest Kodu:** Bulunan test case'leri, OpenAI GPT modeli kullanılarak kapsamlı bir `pytest` dosyasına dönüştürülür.
*   **Kod Analizi:** Radon kütüphanesi ile kod metrikleri (Complexity, Maintainability, LOC vb.) hesaplanır.
*   **Görselleştirme:** Fonksiyon çağrı grafiği ve test sonuçları görselleştirilir.
*   **Kullanıcı Arayüzü:** Streamlit tabanlı kolay kullanımlı bir arayüz sunar.

## Kurulum

Gerekli kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
```

Eğer `gymnasium` hatası alırsanız:
```bash
pip install gymnasium
```

## Kullanım

Uygulamayı başlatmak için:

```bash
streamlit run main.py
```

Tarayıcınızda açılan arayüzden:
1.  Test etmek istediğiniz `.py` dosyasını yükleyin.
2.  OpenAI API anahtarınızı girin (LLM özelliği için gereklidir).
3.  "Test Case Üret" butonuna basarak süreci başlatın.
4.  Sonuçları inceleyin ve üretilen `test_generated.py` dosyasını indirin.

## Sistem Mimarisi

*   **`main.py`**: Streamlit arayüzü ve ana akış kontrolü.
*   **`rl_module.py`**: RL ortamı (`TestGenEnv`) ve PPO ajanı eğitimi. Kod kapsamını (coverage) ödül fonksiyonu olarak kullanır.
*   **`llm_module.py`**: OpenAI API entegrasyonu. Test senaryolarını pytest koduna çevirir.
*   **`analiz_module.py`**: Statik kod analizi (Radon) ve çağrı grafiği oluşturma.

## Algoritma Akışı

1.  Kullanıcı kodu yükler.
2.  RL ajanı, kodun Coverage'ını arttırmak için rastgele veya öğrenilmiş stratejilerle fonksiyon girdileri üretir.
3.  Her girdi için fonksiyon çalıştırılır ve kapsanan satırlar `coverage` kütüphanesi ile ölçülür.
4.  Yeni satırlar kapsayan girdiler "yararlı test case" olarak kaydedilir.
5.  Coverage hedefi (örneğin %85) tutturulana kadar veya maksimum adım sayısına kadar döngü devam eder.
6.  Toplanan test case'ler LLM'e gönderilerek assertion'ları içeren tam bir test dosyası oluşturulur.

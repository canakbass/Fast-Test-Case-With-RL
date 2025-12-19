# FAST TEST CASE
## Otomatik Test Üretim Sistemi
### Kullanım ve Lisans Sözleşmesi

---

## 1. TARAFLAR

**1.1. Tedarikçi Firma:**
- **Unvan:** İNERAI Yazılım Teknolojileri San. ve Tic. Ltd. Şti.
- **Adres:** Teknopark İstanbul, Pendik/İstanbul
- **Vergi No:** 1234567890
- **E-posta:** info@inerai.com
- **Telefon:** +90 216 XXX XX XX

**1.2. Müşteri Firma:**
- **Unvan:** CANSOFTWARE Bilişim Hizmetleri A.Ş.
- **Adres:** Cyberpark Ankara, Bilkent/Ankara
- **Vergi No:** 9876543210
- **E-posta:** info@cansoftware.com
- **Telefon:** +90 312 XXX XX XX

---

## 2. TANIMLAR

**2.1. FAST TEST CASE:** İNERAI tarafından geliştirilmiş, yapay zeka ve pekiştirmeli öğrenme (Reinforcement Learning) tabanlı, otomatik test senaryosu üretim yazılımıdır. Bu sistem, Python kodları için otomatik test case'leri üretir, coverage analizi yapar ve pytest uyumlu test kodları oluşturur. Sistemin modül ve fonksiyon listesi EK-D dokümanında yer almaktadır.

**2.2. Pekiştirmeli Öğrenme Modeli:** FAST TEST CASE sisteminin temelini oluşturan, PPO (Proximal Policy Optimization) algoritması ile eğitilmiş yapay zeka modelidir.

**2.3. Boundary Value Analysis:** Kodun sınır değerlerini tespit eden ve bu değerlere göre test senaryoları üreten analiz modülüdür.

**2.4. Coverage Analysis:** Üretilen test case'lerin kaynak kodu ne oranda kapsadığını ölçen ve raporlayan modüldür.

**2.5. Fine-Tuning:** Temel modelin, müşterinin özel kod yapısına uyarlanması için yapılan ek eğitim sürecidir.

**2.6. Kurulum:** FAST TEST CASE sisteminin CANSOFTWARE tarafından temin edilen sunucular üzerinde çalışır hale getirilmesi işlemidir.

**2.7. Eğitim:** FAST TEST CASE sisteminin etkin kullanımı için CANSOFTWARE personelinin bilgilendirilmesi ve eğitilmesi çalışmasıdır.

---

## 3. KURULUM ve UYARLAMA

**3.1. Sistemin Uyarlanması:** CANSOFTWARE, sözleşme kapsamında EK-D'de yer alan tüm modülleri ve işlevleri satın almaktadır. Bu kapsamda İNERAI; sistemin CANSOFTWARE'in yazılım geliştirme süreçlerine uygun şekilde yapılandırılmasından sorumludur.

**3.2. Sistem İhtiyaçları:** FAST TEST CASE'in çalışması için gerekli sistem yapılanması ve bu yapılanmada ihtiyaç duyulan donanım, işletim sistemi ve ek yazılımlar EK-A ve EK-B'de belirlenmiştir. İNERAI'ın önerdiği bu yazılım ve donanım bileşenleri ile birlikte sistem, 100 kullanıcıya kadar 3 yıl boyunca donanımsal bir güncelleme gerekmeksizin sorunsuz olarak çalışacaktır.

**3.3. Sistem Kurulumu:** FAST TEST CASE'in çalışması için gerekli sistem yapılanmasının CANSOFTWARE tarafından sağlanmasıdır. İNERAI bu yapılanma sırasında CANSOFTWARE'e danışmanlık yapacaktır.

**3.4. Yazılım Kurulumu:** FAST TEST CASE'in çalışması için gerekli sistem yapılanmasının sağlanmasından sonra İNERAI tarafından 7 (yedi) iş günü içerisinde gerçekleştirilir.

**3.5. Kurulum ve Sonrasında Karşılaşılabilecek Sorunlar ve Çözümü:**

**3.5.1. Karşılaşılabilecek Sorunlar:**
- 3.5.1.1. FAST TEST CASE'in kurulduğu donanım üzerinde yaşanabilecek hard-disk, sistem kartı, işlemci, RAM, GPU arızası v.b. olası arızalar.
- 3.5.1.2. FAST TEST CASE'in kurulduğu donanım üzerinde çalışacak işletim sistemi, Python ortamı ve bağımlılıklarda var olması olası hatalar.
- 3.5.1.3. FAST TEST CASE'in kurulduğu donanımların ağ altyapısında meydana gelebilecek kopmalar ve performans düşüşleri.
- 3.5.1.4. FAST TEST CASE'in kod yapısındaki yazılım hataları veya öngörülemeyen durumlar.

**3.5.2. Sorunların Çözümü:**
- 3.5.2.1. FAST TEST CASE'in kod yapısı ile ilgili sorunların giderilmesinden İNERAI, sistemsel yapının işlevselliğinden CANSOFTWARE birinci derecede sorumludur.
- 3.5.2.2. İNERAI; FAST TEST CASE'in yer aldığı sistemin işlevselliğinin sağlanması ve sürdürülmesi konusunda CANSOFTWARE'e danışmanlık yapacaktır.
- 3.5.2.3. İNERAI; kurulum sırasında ve sonrasında CANSOFTWARE sisteminde gördüğü açıkları ve riskleri yazılı olarak bildirecektir.

**3.5.3. Çözümsüzlük Durumu:**
- 3.5.3.1. CANSOFTWARE; ürün kabulünü takip eden 1 (bir) yıl içerisinde FAST TEST CASE'in işlevselliğini sağlamada İNERAI'a aktif görev verdiği halde çözüm elde edemez ise ürün iadesine gidebilir. Bu durumda İNERAI, CANSOFTWARE'in ödediği lisans bedelini, ürünün satın alındığı tarihteki Euro bedeli karşılığını, ürün iade tarihinden sonraki 1 (bir) ay içerisinde CANSOFTWARE'e nakden ve defaten geri öder.

---

## 4. MODEL EĞİTİMİ ve FINE-TUNING

**4.1. Temel Model:** İNERAI, FAST TEST CASE sisteminin temel pekiştirmeli öğrenme modelini eğitilmiş olarak teslim edecektir.

**4.2. Fine-Tuning Hizmeti:** CANSOFTWARE'in özel kod yapılarına uygun fine-tuning işlemi, kurulum sonrasında İNERAI tarafından gerçekleştirilecektir.

**4.3. Fine-Tuning Süresi:** Her bir kod modülü için fine-tuning işlemi en fazla 30.000 adım (step) sürecektir. İşlem süresi donanım kapasitesine bağlı olarak değişkenlik gösterebilir.

**4.4. Model Güncelleme:** İNERAI, destek süresi içerisinde temel modeli iyileştiren güncellemeleri CANSOFTWARE'e ücretsiz olarak sağlayacaktır.

---

## 5. EĞİTİM

FAST TEST CASE sisteminin kullanımı ile ilgili olarak kurulumun tamamlanmasını müteakip, İNERAI iki aşamalı bir eğitim uygulayacaktır.

**5.1. Yazılım Geliştirici Eğitimi:** CANSOFTWARE'in yazılım geliştiricilerinin tamamına eğitim sağlanacaktır. Eğitim süresi en az 3 (üç) iş günü olacaktır. Programın garanti süresi içinde personel yeni eğitime ihtiyaç duyarsa İNERAI bu eğitimi ücretsiz vermeyi kabul ve taahhüt eder.

**5.2. QA/Test Mühendisi Eğitimi:** Test ekibinin tamamına eğitim sağlanacaktır. Eğitim süresi en az 2 (iki) iş günü olacaktır. Programın garanti süresi içinde personel yeni eğitime ihtiyaç duyarsa İNERAI bu eğitimi ücretsiz vermeyi kabul ve taahhüt eder.

**5.3. Eğitim Yeri:** Eğitim yeri CANSOFTWARE tarafından belirlenecek ve sağlanacaktır. Online eğitim de tercih edilebilir.

**5.4. Eğitim Materyalleri:** Eğitim için hazırlanan kullanıcı dökümanları web tabanlı bir yapı üzerinden kullanıcılara açılacaktır.

---

## 6. TESLİMAT

**6.1. Ürün Teslimi:** İNERAI, FAST TEST CASE sisteminin kurulumunun tamamlandığını CANSOFTWARE'e sözleşme imza tarihinden itibaren 30 (otuz) gün içerisinde yazılı olarak bildirecektir.

**6.2. Ürün Testi:** Ürün teslim bildiriminin ardından CANSOFTWARE, EK-D'de belirtilen tüm fonksiyon ve ekranları çalıştırıp test edecek ve onaylayacaktır.

**6.3. Ürün Kabulü:** CANSOFTWARE'in FAST TEST CASE sisteminin; Madde 3'te belirtilen kurulumu, Madde 5'te belirtilen eğitimi kapsayan çalışmaların tamamlandığını ve Madde 6.2'de belirtilen şekliyle test işlemini onaylamasıyla ürün kabulü yapılacaktır.

**6.4. Kabul Süresi:** Kabul işlemi, İNERAI'ın ürün teslimini yazılı bildirmesinden sonraki 10 (on) iş günü içinde tamamlanacaktır.

---

## 7. DESTEK ve GÜVENLİK

**7.1. Destek Süresi:** İNERAI, FAST TEST CASE sistemini CANSOFTWARE'in kabul onayından sonraki 1 (bir) yıl süre ile ücretsiz destekleyecektir. Bu süre CANSOFTWARE'in talebi doğrultusunda uzatılabilir.

**7.2. Destek Kapsamı:** FAST TEST CASE sisteminin içerebileceği tüm yazılım hataları ve hatalı işleyişlerin en kısa süre zarfında düzeltilmesi ile sistemin yenilikleri kapsamaktadır. Bu yenilikler; Python sürüm güncellemeleri, yeni test framework desteği ve performans iyileştirmelerini de kapsar.

**7.3. Destek Şekli ve Zamanlama:** İNERAI, CANSOFTWARE'in e-posta ve telefon yoluyla arıza bildirimini takip eden 4 (dört) saat içerisinde; telefon, uzaktan erişim veya yerinde destek yöntemlerinden uygun olanı ile ilk müdahaleyi gerçekleştirecektir.

- İNERAI ilk 4 saat içinde müdahale etmezse, geciken her saat için 50 Euro tazminat ödeyecektir.
- Süresinde müdahale edilip de 48 saat içinde arızanın giderilememesi halinde İNERAI her saat başına 5 Euro tazminat ödeyecektir.
- İşbu tazminat tutarı 7 gün içinde CANSOFTWARE'e nakden ödenir veya İNERAI'ın alacağından mahsup edilir.

**7.4. Güvenlik:** FAST TEST CASE sisteminin güvenliği için gerekli sistem yapılanması İNERAI tarafından EK-B ve EK-C'de tanımlanmıştır. İNERAI; CANSOFTWARE'in bu yapılanmaya uyup uymadığını belirli periyotlarla inceler ve eksiklikleri yazılı olarak bildirir.

---

## 8. LİSANSLAMA

**8.1.** CANSOFTWARE, FAST TEST CASE sisteminin 100 kullanıcı lisansına süresiz sahip olacaktır. İlerleyen yıllarda kullanıcı sayısının artması durumunda aşağıdaki farklar ödenecektir:

| Kullanıcı Sayısı | Lisans Fark Bedeli |
|------------------|-------------------|
| 101 – 250        | 5,000 Euro + KDV  |
| 251 – 500        | 10,000 Euro + KDV |
| 501 – 1000       | 20,000 Euro + KDV |

**8.2.** İNERAI, FAST TEST CASE sisteminin üreticisi, tüm fikri ve mülki haklarının sahibidir.

**8.3.** CANSOFTWARE, İNERAI'ın FAST TEST CASE sistemi için verdiği kullanıcı lisansı ile mevcut uygulamaları değiştiremez ve yeni uygulamalar geliştiremez. Kaynak koduna müdahale edemez.

**8.4.** İNERAI tarafından verilen FAST TEST CASE sistemi kullanıcı lisans hakları üçüncü şahıslara kullandırılamaz, kiralanamaz ve devredilemez.

---

## 9. GİZLİLİK

**9.1.** Taraflar sözleşmenin devamı süresince, taraflar ile ilgili olarak öğrenilecek bilgileri GİZLİ bilgi olarak kabul etmişlerdir. Gizli bilgiler, gerekçesi ne olursa olsun üçüncü kişilere açıklanamaz.

**9.2.** Taraflar sözleşmenin devamı süresince ve sonrasında diğer tarafın yazılı onayı olmadıkça aşağıdaki gizli bilgileri üçüncü kişilere açıklayamaz:

- 9.2.1. İNERAI'a ait yazılım kaynak kodları, algoritma yapıları, model ağırlıkları, eğitim verileri ve know-how bilgileri.
- 9.2.2. CANSOFTWARE'in iç yapısı, organizasyonu, ticari bilgileri ve müşteri verileri.
- 9.2.3. CANSOFTWARE'in FAST TEST CASE ile analiz ettiği tüm kaynak kodlar ve test sonuçları.

**9.3.** Taraflar sözleşmenin herhangi bir sebepten sona ermesi hallerinde dahi 5 (beş) yıl süre ile gizlilik hükmüne uymakla yükümlüdür.

---

## 10. SÖZLEŞME BEDELİ VE ÖDEME

**10.1.** FAST TEST CASE sisteminin CANSOFTWARE için 100 kullanıcılı lisans bedeli **15,000 EURO + KDV**'dir. Sözleşme kapsamında eğitimler, fine-tuning ve Madde 7'de tanımlanan destek hizmeti verilecektir. Ödemelerin TL yapılması durumunda TCMB döviz satış kuru esas alınacaktır.

**10.2.** CANSOFTWARE, Madde 10.1'de tanımlanan sözleşme bedelinin %30'unu (4,500 Euro) sözleşme tarihinde İNERAI'a nakden öder. Ürünün kabul edilmemesi halinde peşin ödenen bedel 7 gün içerisinde CANSOFTWARE'e iade edilecektir.

**10.3.** Kalan miktar (10,500 Euro) CANSOFTWARE'in Madde 6.3'te tanımlanan ürün kabulünü yapmasının ardından İNERAI'a 5 eşit aylık taksitle (2,100 Euro/ay) ödenir.

**10.4.** FAST TEST CASE sisteminin bir yıllık ücretsiz destek süresi sona erdikten sonra, Madde 7'de tanımlanan destek hizmeti yıllık lisans bedelinin %15'i (2,250 Euro) karşılığında 1'er yıllık dönemler halinde uzatılır.

---

## 11. SÖZLEŞMENİN SONA ERMESİ

Bu sözleşmenin devam ettiği süre zarfında İNERAI firmasının iflas etmesi veya herhangi bir sebeple yazılım geliştirme ve destekleme faaliyetlerine devam edememesi durumunda, FAST TEST CASE programına ait tüm kaynak kodlarını ve dokümantasyonu CANSOFTWARE'e hiçbir ücret talep etmeden devretmeyi kabul ve taahhüt eder.

---

## 12. TEBLİGAT

İşbu sözleşmenin uygulanması süresince tarafların bildirim ve tebligat için kullanacakları isim ve adresler sözleşmenin 1. maddesinde tanımlanmıştır. Taraflar bu adreslerde vaki olacak değişiklikleri diğer tarafa yazılı olarak bildirmedikleri takdirde, eski adreslerine yapılacak tebligatların geçerli sayılacağını kabul eder.

---

## 13. UYUŞMAZLIKLARIN ÇÖZÜMÜ

İşbu sözleşme ve uygulanması sebebiyle ortaya çıkabilecek her türlü uyuşmazlığı Taraflar öncelikle kendi aralarında sulhen halletme yoluna gideceklerdir. Uyuşmazlığın kendi aralarında giderilmesinin mümkün olmadığı durumlarda, İstanbul Mahkemeleri ve İcra Daireleri yetkilidir.

---

## 14. İMZA

İşbu sözleşme 14 (ondört) madde ve 4 (dört) ekten ibaret olup, taraflarca okunmuş ve kabul edilerek ..../..../2025 tarihinde 2 (iki) nüsha olarak imza altına alınmıştır.

| **İNERAI Yazılım Teknolojileri** | **CANSOFTWARE Bilişim Hizmetleri** |
|----------------------------------|-----------------------------------|
| Yetkili: ........................ | Yetkili: ........................ |
| İmza: ........................... | İmza: ........................... |
| Tarih: ..../..../2025           | Tarih: ..../..../2025            |
| Kaşe:                            | Kaşe:                             |

---

# EKLER

## EK-A: Donanım Gereksinimleri

| Bileşen | Minimum | Önerilen |
|---------|---------|----------|
| İşlemci | Intel Core i5 / AMD Ryzen 5 | Intel Core i7 / AMD Ryzen 7 |
| RAM | 8 GB | 16 GB |
| Disk | 50 GB SSD | 100 GB NVMe SSD |
| GPU | - | NVIDIA RTX 3060 (Fine-tuning için) |
| İşletim Sistemi | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |

---

## EK-B: Yazılım Gereksinimleri

| Yazılım | Sürüm |
|---------|-------|
| Python | 3.9 veya üzeri |
| PyTorch | 2.0+ |
| Stable-Baselines3 | 2.0+ |
| Gymnasium | 0.29+ |
| Coverage.py | 7.0+ |
| OpenAI API | Güncel |
| Streamlit | 1.30+ |

---

## EK-C: Güvenlik Gereksinimleri

1. **Ağ Güvenliği:**
   - FAST TEST CASE sunucusu, güvenlik duvarı arkasında çalıştırılmalıdır.
   - HTTPS protokolü zorunludur.
   - API anahtarları (OpenAI vb.) güvenli ortam değişkenlerinde saklanmalıdır.

2. **Erişim Kontrolü:**
   - Kullanıcı kimlik doğrulama sistemi aktif olmalıdır.
   - Rol bazlı yetkilendirme uygulanmalıdır.

3. **Yedekleme:**
   - Model dosyaları ve konfigürasyonlar günlük yedeklenmelidir.
   - Fine-tuned modeller ayrı bir depolama alanında saklanmalıdır.

---

## EK-D: Modüller ve Fonksiyonlar

### 1. RL Test Case Generator Modülü
- PPO tabanlı test case üretim motoru
- Otomatik parametre keşfi
- Coverage-based reward sistemi
- Efficiency-based optimizasyon
- Early stopping mekanizması

### 2. Boundary Value Analysis Modülü
- Kod parsing ve AST analizi
- Koşul ifadesi tespiti (if, while, for)
- Kritik değer belirleme
- Sınır değer test case üretimi

### 3. Code Coverage Modülü
- Satır bazlı coverage ölçümü
- Branch coverage analizi
- Gerçek zamanlı coverage takibi
- Coverage raporlama

### 4. LLM Integration Modülü
- OpenAI GPT entegrasyonu
- Pytest kod üretimi
- Test açıklaması oluşturma
- Kod optimizasyon önerileri

### 5. Fine-Tuning Modülü
- Kod bazlı model uyarlama
- Transfer learning desteği
- İlerleme takibi ve raporlama
- Model kaydetme ve yükleme

### 6. Web Arayüzü (Streamlit)
- Dosya yükleme
- Gerçek zamanlı analiz görüntüleme
- Test case listesi
- Coverage metrikleri
- Pytest kodu indirme

### 7. Raporlama Modülü
- Test sonuç raporları
- Coverage grafikleri
- Exception listesi
- Performans metrikleri

---

**SÖZLEŞME SONU**

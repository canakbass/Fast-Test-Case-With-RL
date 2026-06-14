def ortalama_hesapla(vize, final):
    """
    Vize (%40) ve Final (%60) notuna göre ortalamayı hesaplar.
    """
    sonuc = (vize * 0.4) + (final * 0.6)
    return sonuc

def harf_notu_belirle(ortalama):
    """
    Sayısal ortalamayı alır, harf notuna çevirir.
    """
    if ortalama >= 90:
        return "AA (Mükemmel)"
    elif ortalama >= 85:
        return "BA (Pekiyi)"
    elif ortalama >= 70:
        return "BB (İyi)"
    elif ortalama >= 50:
        return "CC (Geçer)"
    else:
        return "FF (Kaldı)"

def basari_durumu_yazdir(isim, vize_notu, final_notu):
    """
    BU ANA FONKSİYONDUR.
    Diğer fonksiyonları çağırarak sonuç üretir.
    """
    # 1. Adım: Diğer fonksiyonu çağırıp ortalamayı alıyoruz
    ort = ortalama_hesapla(vize_notu, final_notu)
    
    # 2. Adım: Hesaplanan ortalamayı diğer fonksiyona gönderip harfi alıyoruz
    harf = harf_notu_belirle(ort)
    
    # 3. Adım: Sonuçları ekrana basıyoruz
    print(f"-" * 30)
    print(f"Öğrenci Adı: {isim}")
    print(f"Ortalama   : {ort:.2f}")
    print(f"Harf Notu  : {harf}")
    print(f"-" * 30)

# --- Programı Başlatma ---
# Sadece ana fonksiyonu çağırmamız yeterlidir, o diğerlerini halleder.
basari_durumu_yazdir("Ahmet Yılmaz", 60, 80)
basari_durumu_yazdir("Ayşe Demir", 45, 50)
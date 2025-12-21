"""
White Box Test Cases
====================
Bu dosya, projenin white box testlerini içerir.
- Selenium ile web arayüzü testi
- pytest ile unit testler

Kullanım:
    pytest tests/test_whitebox.py -v
    
    (Selenium testi için önce Streamlit'i başlatın:)
    streamlit run main.py
"""

import pytest
import time
import os
import sys

# Proje kök dizinini path'e ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from boundary_analysis import BoundaryValueExtractor, extract_boundary_values, calculate_boundary_reward
from rl_module_v2 import extract_code_features
from analiz_module import get_metrics


# ============================================================================
# TEST CASE 1: Boundary Analysis Modülü White Box Testi
# ============================================================================
class TestBoundaryAnalysis:
    """
    White Box Test - Boundary Value Analysis Modülü
    
    Bu test, boundary_analysis.py modülündeki fonksiyonların
    iç mantığını (internal logic) test eder.
    
    Test edilen kod yolları:
    - AST parsing
    - Karşılaştırma operatörlerinin tespiti
    - Sınır değerlerinin hesaplanması
    """
    
    def test_extract_boundary_values_with_comparisons(self):
        """
        Test Case 1.1: Karşılaştırma operatörlerinin tespiti
        
        White Box Yaklaşım:
        - Kodda bulunan '<', '>', '<=', '>=' operatörlerini tespit eder
        - Her operatör için doğru sınır değerlerini üretir
        """
        # Arrange - Test edilecek kod
        code = '''
def check_age(age):
    if age < 18:
        return "child"
    elif age >= 65:
        return "senior"
    else:
        return "adult"
'''
        
        # Act - Sınır değerlerini çıkar
        result = extract_boundary_values(code)
        
        # Assert - Sonuçları doğrula
        assert 'boundaries' in result, "Sonuçta 'boundaries' anahtarı olmalı"
        assert 'conditions' in result, "Sonuçta 'conditions' anahtarı olmalı"
        
        # En az 2 koşul bulunmalı (age < 18 ve age >= 65)
        assert len(result['conditions']) >= 2, f"En az 2 koşul bekleniyor, {len(result['conditions'])} bulundu"
        
        # Kritik değerler kontrol
        all_boundary_values = []
        for b in result['boundaries']:
            all_boundary_values.extend(b.get('boundary_values', []))
        
        # 18 ve 65 sınır değerleri bulunmalı
        assert 18 in all_boundary_values or 17 in all_boundary_values, "18 sınır değeri bekleniyor"
        assert 65 in all_boundary_values or 64 in all_boundary_values, "65 sınır değeri bekleniyor"
        
        print(f"✓ Bulunan koşullar: {result['conditions']}")
        print(f"✓ Sınır değerleri: {all_boundary_values}")
    
    def test_boundary_reward_calculation(self):
        """
        Test Case 1.2: Boundary Reward hesaplama
        
        White Box Yaklaşım:
        - Farklı input değerleri için reward hesaplamasını test eder
        - Sınır değerine yakın inputlarda yüksek reward beklenir
        """
        # Arrange
        code = '''
def is_adult(age):
    if age >= 18:
        return True
    return False
'''
        
        # Act & Assert - Sınır değerine yakın inputlar
        # age=18 sınır değeri, bu yüzden reward yüksek olmalı
        reward_at_boundary = calculate_boundary_reward([18], code)
        reward_far_from_boundary = calculate_boundary_reward([100], code)
        
        # Sınır değerinde reward, uzak değerden yüksek olmalı
        assert reward_at_boundary >= reward_far_from_boundary, \
            f"Sınır değerinde reward ({reward_at_boundary}) >= uzak değer reward ({reward_far_from_boundary}) olmalı"
        
        print(f"✓ Sınır değeri (18) reward: {reward_at_boundary}")
        print(f"✓ Uzak değer (100) reward: {reward_far_from_boundary}")
    
    def test_ast_visitor_operators(self):
        """
        Test Case 1.3: AST Visitor operatör tanıma
        
        White Box Yaklaşım:
        - Tüm karşılaştırma operatörlerinin doğru tanındığını test eder
        """
        # Arrange - Tüm operatörleri içeren kod
        code = '''
def test_operators(a, b, c, d, e):
    if a < 10:
        pass
    if b > 20:
        pass
    if c <= 30:
        pass
    if d >= 40:
        pass
    if e == 50:
        pass
'''
        
        # Act
        result = extract_boundary_values(code)
        
        # Assert - Her operatör için koşul bulunmalı
        operators_found = [b['operator'] for b in result['boundaries']]
        
        assert '<' in operators_found, "< operatörü bulunamadı"
        assert '>' in operators_found, "> operatörü bulunamadı"
        assert '<=' in operators_found, "<= operatörü bulunamadı"
        assert '>=' in operators_found, ">= operatörü bulunamadı"
        assert '==' in operators_found, "== operatörü bulunamadı"
        
        print(f"✓ Bulunan operatörler: {operators_found}")


# ============================================================================
# TEST CASE 2: RL Module Feature Extraction White Box Testi
# ============================================================================
class TestRLModuleFeatureExtraction:
    """
    White Box Test - RL Module Kod Analizi
    
    Bu test, rl_module_v2.py'deki extract_code_features fonksiyonunun
    iç mantığını test eder.
    
    Test edilen kod yolları:
    - AST node sayımı
    - Feature vektör boyutu
    - Normalizasyon
    """
    
    def test_feature_extraction_dimensions(self):
        """
        Test Case 2.1: Feature vektör boyutu
        
        White Box Yaklaşım:
        - Observation space için sabit 20 boyutlu vektör üretilmeli
        """
        # Arrange
        code = '''
def simple_func(x):
    return x * 2
'''
        
        # Act
        features = extract_code_features(code)
        
        # Assert
        assert len(features) == 20, f"Feature vektörü 20 boyutlu olmalı, {len(features)} bulundu"
        assert features.dtype.name.startswith('float'), "Feature vektörü float olmalı"
        
        print(f"✓ Feature vektör boyutu: {len(features)}")
        print(f"✓ Feature değerleri: {features[:5]}... (ilk 5)")
    
    def test_feature_extraction_if_counting(self):
        """
        Test Case 2.2: If statement sayımı
        
        White Box Yaklaşım:
        - Koddaki if sayısının doğru tespit edilmesi
        - Complexity estimate'in if sayısına göre artması
        """
        # Arrange - Farklı sayıda if içeren kodlar
        code_no_if = '''
def no_branch(x):
    return x
'''
        
        code_with_ifs = '''
def with_branches(x):
    if x > 0:
        if x > 10:
            return "big"
        return "positive"
    if x < 0:
        return "negative"
    return "zero"
'''
        
        # Act
        features_no_if = extract_code_features(code_no_if)
        features_with_ifs = extract_code_features(code_with_ifs)
        
        # Assert - if içeren kodun karmaşıklığı daha yüksek olmalı
        # Feature vektöründeki değerler normalize edilmiş
        # Toplam değer karşılaştırması yapabiliriz
        complexity_no_if = features_no_if.sum()
        complexity_with_ifs = features_with_ifs.sum()
        
        assert complexity_with_ifs > complexity_no_if, \
            f"If içeren kod daha karmaşık olmalı: {complexity_with_ifs} > {complexity_no_if}"
        
        print(f"✓ If'siz kod karmaşıklığı: {complexity_no_if:.4f}")
        print(f"✓ If'li kod karmaşıklığı: {complexity_with_ifs:.4f}")
    
    def test_feature_extraction_invalid_code(self):
        """
        Test Case 2.3: Geçersiz kod handling
        
        White Box Yaklaşım:
        - Syntax hatası olan kod için sıfır vektör dönmeli
        - Hata fırlatmamalı
        """
        # Arrange - Geçersiz Python kodu
        invalid_code = '''
def broken_func(
    if True
        return
'''
        
        # Act
        features = extract_code_features(invalid_code)
        
        # Assert - Sıfır vektör dönmeli
        assert len(features) == 20, "Geçersiz kod için de 20 boyutlu vektör dönmeli"
        assert features.sum() == 0, "Geçersiz kod için sıfır vektör bekleniyor"
        
        print(f"✓ Geçersiz kod için feature vektörü: {features}")


# ============================================================================
# TEST CASE 3: Analiz Module Metrics White Box Testi
# ============================================================================
class TestAnalyzModuleMetrics:
    """
    White Box Test - Kod Metrikleri Analizi
    
    Bu test, analiz_module.py'deki get_metrics fonksiyonunun
    iç mantığını test eder.
    
    Test edilen kod yolları:
    - LOC (Lines of Code) hesaplama
    - Cyclomatic Complexity hesaplama
    - Maintainability Index hesaplama
    """
    
    def test_metrics_loc_calculation(self):
        """
        Test Case 3.1: LOC (Lines of Code) hesaplama
        
        White Box Yaklaşım:
        - Satır sayısının doğru hesaplanması
        - SLOC (Source Lines of Code) vs LOC farkı
        """
        # Arrange
        code = '''
def hello():
    # Bu bir yorum
    print("Hello")
    
    # Başka yorum
    print("World")
'''
        
        # Act
        metrics = get_metrics(code)
        
        # Assert
        assert 'LOC' in metrics, "LOC metriği bulunamadı"
        assert 'SLOC' in metrics, "SLOC metriği bulunamadı"
        assert 'Comments' in metrics, "Comments metriği bulunamadı"
        
        # SLOC <= LOC olmalı (yorumlar ve boş satırlar hariç)
        assert metrics['SLOC'] <= metrics['LOC'], "SLOC <= LOC olmalı"
        
        print(f"✓ LOC: {metrics['LOC']}")
        print(f"✓ SLOC: {metrics['SLOC']}")
        print(f"✓ Comments: {metrics['Comments']}")
    
    def test_metrics_complexity(self):
        """
        Test Case 3.2: Cyclomatic Complexity hesaplama
        
        White Box Yaklaşım:
        - Branch sayısına göre complexity artmalı
        - Her if/for/while +1 complexity
        """
        # Arrange - Basit fonksiyon
        simple_code = '''
def simple():
    return 1
'''
        
        # Arrange - Karmaşık fonksiyon
        complex_code = '''
def complex_func(x, y):
    if x > 0:
        if y > 0:
            for i in range(x):
                if i % 2 == 0:
                    print(i)
        else:
            while y < 0:
                y += 1
    return x + y
'''
        
        # Act
        simple_metrics = get_metrics(simple_code)
        complex_metrics = get_metrics(complex_code)
        
        # Assert
        assert complex_metrics['Total Complexity'] > simple_metrics['Total Complexity'], \
            "Karmaşık kodun complexity'si daha yüksek olmalı"
        
        print(f"✓ Basit kod complexity: {simple_metrics['Total Complexity']}")
        print(f"✓ Karmaşık kod complexity: {complex_metrics['Total Complexity']}")
    
    def test_metrics_maintainability_index(self):
        """
        Test Case 3.3: Maintainability Index hesaplama
        
        White Box Yaklaşım:
        - MI değeri 0-100 arası olmalı
        - Rank A, B, C olabilir
        """
        # Arrange
        code = '''
def well_documented_func(x: int, y: int) -> int:
    """
    Bu fonksiyon iki sayıyı toplar.
    
    Args:
        x: İlk sayı
        y: İkinci sayı
    
    Returns:
        Toplam
    """
    result = x + y
    return result
'''
        
        # Act
        metrics = get_metrics(code)
        
        # Assert
        assert 'Maintainability Index' in metrics, "MI metriği bulunamadı"
        assert 'MI Rank' in metrics, "MI Rank bulunamadı"
        
        mi = metrics['Maintainability Index']
        assert 0 <= mi <= 100, f"MI 0-100 arası olmalı, {mi} bulundu"
        assert metrics['MI Rank'] in ['A', 'B', 'C'], f"MI Rank A/B/C olmalı, {metrics['MI Rank']} bulundu"
        
        print(f"✓ Maintainability Index: {mi:.2f}")
        print(f"✓ MI Rank: {metrics['MI Rank']}")


# ============================================================================
# BONUS: Selenium Web UI Testi (Opsiyonel)
# ============================================================================
class TestSeleniumWebUI:
    """
    White Box Test - Selenium Web Arayüzü Testi
    
    NOT: Bu test için Streamlit uygulamasının çalışıyor olması gerekir:
    streamlit run main.py
    
    Çalıştırmak için:
    pytest tests/test_whitebox.py::TestSeleniumWebUI -v
    """
    
    @pytest.fixture(scope="class")
    def driver(self):
        """Selenium WebDriver fixture"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service
            from selenium.webdriver.chrome.options import Options
            from webdriver_manager.chrome import ChromeDriverManager
            
            # Chrome options
            chrome_options = Options()
            chrome_options.add_argument("--headless")  # Headless mode
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
            # Driver oluştur
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.implicitly_wait(10)
            
            yield driver
            
            driver.quit()
        except Exception as e:
            pytest.skip(f"Selenium kurulu değil veya Chrome bulunamadı: {e}")
    
    @pytest.mark.skip(reason="Streamlit çalışıyor olmalı - manuel test için skip kaldırın")
    def test_streamlit_page_loads(self, driver):
        """
        Test Case S1: Streamlit sayfası yükleniyor mu?
        
        White Box Yaklaşım:
        - Ana sayfa yüklenmeli
        - Başlık görünmeli
        """
        from selenium.webdriver.common.by import By
        
        # Act
        driver.get("http://localhost:8501")
        time.sleep(3)  # Streamlit yüklenmesi için bekle
        
        # Assert
        assert "Test" in driver.title or "Streamlit" in driver.title, \
            f"Sayfa başlığı beklenen değil: {driver.title}"
        
        # Ana başlık kontrolü
        page_source = driver.page_source
        assert "Reinforcement Learning" in page_source or "Test Case" in page_source, \
            "Ana başlık bulunamadı"
        
        print("✓ Streamlit sayfası başarıyla yüklendi")
    
    @pytest.mark.skip(reason="Streamlit çalışıyor olmalı - manuel test için skip kaldırın")
    def test_file_upload_element_exists(self, driver):
        """
        Test Case S2: Dosya yükleme elementi var mı?
        
        White Box Yaklaşım:
        - File uploader komponenti bulunmalı
        """
        from selenium.webdriver.common.by import By
        
        # Act
        driver.get("http://localhost:8501")
        time.sleep(3)
        
        # Assert - File uploader kontrolü
        page_source = driver.page_source
        assert "Python Dosyası Yükle" in page_source or "file_uploader" in page_source, \
            "Dosya yükleme alanı bulunamadı"
        
        print("✓ Dosya yükleme elementi mevcut")
    
    @pytest.mark.skip(reason="Streamlit çalışıyor olmalı - manuel test için skip kaldırın")
    def test_sidebar_settings_exist(self, driver):
        """
        Test Case S3: Sidebar ayarları var mı?
        
        White Box Yaklaşım:
        - Ayarlar sidebar'ı görünmeli
        - API Key input'u olmalı
        """
        from selenium.webdriver.common.by import By
        
        # Act
        driver.get("http://localhost:8501")
        time.sleep(3)
        
        # Assert - Sidebar kontrolü
        page_source = driver.page_source
        assert "Ayarlar" in page_source or "Settings" in page_source, \
            "Sidebar ayarları bulunamadı"
        
        print("✓ Sidebar ayarları mevcut")


# ============================================================================
# Test Runner
# ============================================================================
if __name__ == "__main__":
    # Testleri çalıştır
    pytest.main([__file__, "-v", "--tb=short"])

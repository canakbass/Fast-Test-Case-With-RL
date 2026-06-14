import streamlit as st
import os
import shutil
import tempfile
import importlib

# Module'leri reload et (geliştirme sırasında cache sorunlarını önler)
import rl_module_v2
import llm_module
import boundary_analysis
importlib.reload(rl_module_v2)
importlib.reload(llm_module)
importlib.reload(boundary_analysis)

from rl_module_v2 import generate_test_cases, train_base_model
from llm_module import generate_pytest_code
from analiz_module import get_metrics, generate_call_graph
from boundary_analysis import analyze_code_for_testing
import graphviz

st.set_page_config(page_title="AI Test Generator", layout="wide")

st.title("🧪 Reinforcement Learning ile Otomatik Test Case Üretimi")
st.caption("Boundary Value Analysis + Coverage Maximization")

# Sidebar
st.sidebar.header("⚙️ Ayarlar")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

st.sidebar.markdown("---")
st.sidebar.subheader("Model Ayarları")
fine_tune_steps = st.sidebar.slider("Fine-Tune Step (Fonksiyon Başına)", 0, 50000, 10000, step=5000)
num_episodes = st.sidebar.slider("Episode Sayısı", 1, 10, 2)  # Azaltıldı: 5-50 -> 1-10, default 2
use_boundary = st.sidebar.checkbox("Boundary Analysis Kullan", value=True)
save_finetuned = st.sidebar.checkbox("Fine-Tuned Modeli Kaydet", value=True)

uploaded_file = st.file_uploader("📂 Python Dosyası Yükle", type="py")

if uploaded_file is not None:
    code_content = uploaded_file.read().decode("utf-8")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Yüklenen Kod")
        st.code(code_content, language="python")

        # Metrics
        if st.button("📊 Kod Analizi Yap"):
            metrics = get_metrics(code_content)
            st.write("### Kod Metrikleri (Radon)")
            st.json(metrics)

            # Boundary Analysis
            if use_boundary:
                st.write("### 🎯 Sınır Değer Analizi")
                analysis = analyze_code_for_testing(code_content)
                
                with st.expander("Bulunan Koşullar", expanded=True):
                    conditions = analysis['boundary_analysis'].get('conditions', [])
                    if conditions:
                        for cond in conditions:
                            st.write(f"• `{cond}`")
                    else:
                        st.write("Koşul bulunamadı")
                
                with st.expander("Kritik Test Değerleri"):
                    critical = analysis['boundary_analysis'].get('critical_values', [])
                    st.write(f"**Değerler:** {critical[:15]}...")
                
                with st.expander("Önerilen Test Stratejisi"):
                    for strategy in analysis.get('suggested_test_strategy', []):
                        st.write(f"• {strategy}")

            # Graph
            try:
                dot = generate_call_graph(code_content)
                st.write("### 🔗 Fonksiyon Çağrı Grafiği")
                st.graphviz_chart(dot)
            except Exception as e:
                st.warning(f"Grafik oluşturulamadı: {e}")

    with col2:
        st.subheader("🤖 Otomatik Test Üretimi")
        mode = st.radio("Test Case Üretim Modu", ["Hazır Base Modelle Üret", "Fine-Tune Edip Üret"])
        base_model_path = "ppo_testgen_base.zip"

        if st.button("🚀 Test Case Üret", type="primary"):
            # Model var mı kontrol et
            finetuned_model_path = "ppo_testgen_finetuned.zip"
            
            # Fine-tune modunda: ÖNCE base model kullan, fine-tune yap, SONRA kaydet
            # Bir sonraki sefer finetuned model varsa direkt kullan
            if mode == "Fine-Tune Edip Üret":
                # Eğer finetuned model varsa VE fine_tune_steps=0 ise: direkt finetuned kullan
                # Eğer fine_tune_steps > 0 ise: base'den başla, yeniden fine-tune yap
                if fine_tune_steps == 0 and os.path.exists(finetuned_model_path):
                    model_to_use = finetuned_model_path
                    st.success(f"✅ Kaydedilmiş fine-tuned model kullanılıyor!")
                elif os.path.exists(base_model_path):
                    model_to_use = base_model_path
                    if fine_tune_steps > 0:
                        st.info(f"🎯 Base model üzerinde {fine_tune_steps} step fine-tuning yapılacak...")
                else:
                    st.error(f"❌ Model bulunamadı: {base_model_path}")
                    st.info("💡 Önce 'python train_v2.py' ile base modeli eğitin.")
                    model_to_use = None
            else:
                # Hazır model modunda: sadece base model kullan
                if os.path.exists(base_model_path):
                    model_to_use = base_model_path
                    st.info("📦 Base model kullanılıyor (fine-tuning yok)")
                else:
                    st.error(f"❌ Base model bulunamadı: {base_model_path}")
                    st.info("💡 Önce 'python train_v2.py' ile base modeli eğitin.")
                    model_to_use = None
            
            if model_to_use:
                ft_steps = fine_tune_steps if mode == "Fine-Tune Edip Üret" else 0
                
                progress_placeholder = st.empty()
                with st.spinner("Test case'ler üretiliyor..."):
                    cases, info = generate_test_cases(
                        code_content, 
                        model_path=model_to_use,
                        num_episodes=num_episodes,
                        fine_tune_steps=ft_steps,
                        use_boundary_analysis=use_boundary,
                        save_finetuned_model=save_finetuned if mode == "Fine-Tune Edip Üret" else False,
                        progress_callback=lambda msg: progress_placeholder.info(msg)
                    )

                # Sonuç metrikleri
                st.success(f"✅ {len(cases)} adet test senaryosu üretildi!")
                
                # Coverage göstergesi
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Coverage", f"{info['coverage_pct']:.1f}%")
                with col_b:
                    st.metric("Satır", f"{info['coverage_lines']}/{info['total_lines']}")
                with col_c:
                    exc_count = len(info.get('exceptions_found', []))
                    st.metric("Exception", exc_count)
                
                # Exception'lar
                if info.get('exceptions_found'):
                    st.warning(f"🔥 Bulunan Exception'lar: {', '.join(info['exceptions_found'])}")
                
                # Boundary conditions
                if info.get('boundary_conditions'):
                    with st.expander("🎯 Test Edilen Koşullar"):
                        for cond in info['boundary_conditions'][:10]:
                            st.write(f"• `{cond}`")

                st.write("### 📋 Üretilen Test Case'ler")
                # DataFrame için düzenle
                display_cases = []
                for c in cases:
                    display_cases.append({
                        'Input': str(c['input']),
                        'New Lines': len(c['new_lines']),
                        'Exception': c.get('exception', '-'),
                        'Boundary': '✓' if c.get('boundary_hit') else '-',
                        'Reward': c.get('total_reward', c.get('coverage_increase', 0))
                    })
                st.dataframe(display_cases, width='stretch')

                if cases:
                    with st.spinner("LLM ile Pytest koduna çevriliyor..."):
                        pytest_code = generate_pytest_code(cases, code_content)
                        st.write("### 📝 Oluşturulan Pytest Kodu")
                        st.code(pytest_code, language="python")

                        st.download_button(
                            label="⬇️ Test Kodunu İndir",
                            data=pytest_code,
                            file_name="test_generated.py",
                            mime="text/x-python"
                        )
                        
                        # ========== YÜKLENEN KODUN ANALİZİ ==========
                        st.markdown("---")
                        st.subheader("📊 Yüklenen Kodun Analizi")
                        
                        # Yüklenen kod metrikleri
                        test_metrics = get_metrics(code_content)
                        
                        col_m1, col_m2 = st.columns(2)
                        
                        with col_m1:
                            st.write("### 📈 Kod Metrikleri (Radon)")
                            # 10 adet metrik göster
                            metrics_display = {
                                "1. LOC (Total Lines)": test_metrics.get('LOC', 0),
                                "2. SLOC (Source Lines)": test_metrics.get('SLOC', 0),
                                "3. LLOC (Logical Lines)": test_metrics.get('LLOC', 0),
                                "4. Comments": test_metrics.get('Comments', 0),
                                "5. Blank Lines": test_metrics.get('Blank lines', 0),
                                "6. Cyclomatic Complexity": test_metrics.get('Total Complexity', 0),
                                "7. Max Complexity": test_metrics.get('Max Complexity', 0),
                                "8. Maintainability Index": round(test_metrics.get('Maintainability Index', 0), 2),
                                "9. MI Rank": test_metrics.get('MI Rank', 'N/A'),
                                "10. Halstead Volume": round(test_metrics.get('Halstead Volume', 0), 2) if test_metrics.get('Halstead Volume') else 'N/A'
                            }
                            for metric_name, metric_value in metrics_display.items():
                                st.write(f"**{metric_name}:** {metric_value}")
                        
                        with col_m2:
                            st.write("### 🔗 Fonksiyon Çağrı Grafiği")
                            try:
                                test_graph = generate_call_graph(code_content)
                                st.graphviz_chart(test_graph)
                            except Exception as e:
                                st.warning(f"Graf oluşturulamadı: {e}")
                        
                        # Halstead detayları
                        with st.expander("📐 Halstead Metrikleri (Detay)"):
                            halstead_metrics = {
                                "Difficulty": test_metrics.get('Halstead Difficulty', 'N/A'),
                                "Effort": test_metrics.get('Halstead Effort', 'N/A'),
                                "Time (seconds)": test_metrics.get('Halstead Time', 'N/A'),
                                "Estimated Bugs": test_metrics.get('Halstead Bugs', 'N/A')
                            }
                            for k, v in halstead_metrics.items():
                                if v != 'N/A' and isinstance(v, (int, float)):
                                    st.write(f"**{k}:** {round(v, 4)}")
                                else:
                                    st.write(f"**{k}:** {v}")
                        
                        # ========== SONARQUBE BENZERİ KALİTE RAPORU ==========
                        st.markdown("---")
                        st.subheader("🔍 Kod Kalite Raporu (SonarQube Tarzı)")
                        
                        # Kalite skorları hesapla
                        mi = test_metrics.get('Maintainability Index', 0)
                        total_complexity = test_metrics.get('Total Complexity', 0)
                        loc = test_metrics.get('LOC', 1)
                        sloc = test_metrics.get('SLOC', 1)
                        comments = test_metrics.get('Comments', 0)
                        
                        # Maintainability skoru (0-100)
                        maintainability_score = min(100, max(0, mi))
                        
                        # Complexity skoru (daha düşük = daha iyi)
                        complexity_per_loc = total_complexity / max(sloc, 1)
                        if complexity_per_loc < 0.1:
                            complexity_grade = "A"
                            complexity_color = "green"
                        elif complexity_per_loc < 0.2:
                            complexity_grade = "B"
                            complexity_color = "blue"
                        elif complexity_per_loc < 0.3:
                            complexity_grade = "C"
                            complexity_color = "orange"
                        else:
                            complexity_grade = "D"
                            complexity_color = "red"
                        
                        # Comment ratio
                        comment_ratio = (comments / max(loc, 1)) * 100
                        
                        # Genel skor
                        overall_score = (maintainability_score * 0.4 + 
                                        (100 - min(total_complexity * 5, 100)) * 0.3 +
                                        min(comment_ratio * 5, 100) * 0.3)
                        
                        if overall_score >= 80:
                            overall_grade = "A"
                            overall_emoji = "✅"
                        elif overall_score >= 60:
                            overall_grade = "B"
                            overall_emoji = "👍"
                        elif overall_score >= 40:
                            overall_grade = "C"
                            overall_emoji = "⚠️"
                        else:
                            overall_grade = "D"
                            overall_emoji = "❌"
                        
                        # Görsel kartlar
                        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
                        
                        with col_q1:
                            st.metric(
                                label="🏆 Genel Kalite",
                                value=f"{overall_grade} {overall_emoji}",
                                delta=f"{overall_score:.1f}/100"
                            )
                        
                        with col_q2:
                            st.metric(
                                label="🔧 Maintainability",
                                value=test_metrics.get('MI Rank', 'N/A'),
                                delta=f"{maintainability_score:.1f}"
                            )
                        
                        with col_q3:
                            st.metric(
                                label="🔀 Complexity",
                                value=complexity_grade,
                                delta=f"{total_complexity} total"
                            )
                        
                        with col_q4:
                            st.metric(
                                label="💬 Comment Ratio",
                                value=f"{comment_ratio:.1f}%",
                                delta=f"{comments} satır"
                            )
                        
                        # Detaylı rapor
                        with st.expander("📋 Detaylı Kalite Raporu"):
                            st.markdown(f"""
**Kod Kalite Özeti:**
- **Toplam Satır (LOC):** {loc}
- **Kaynak Kod Satırı (SLOC):** {sloc}
- **Mantıksal Satır (LLOC):** {test_metrics.get('LLOC', 0)}
- **Yorum Satırı:** {comments}
- **Boş Satır:** {test_metrics.get('Blank lines', 0)}

**Karmaşıklık Analizi:**
- **Toplam Cyclomatic Complexity:** {total_complexity}
- **Maksimum Complexity:** {test_metrics.get('Max Complexity', 0)}
- **Complexity/SLOC Oranı:** {complexity_per_loc:.3f}
- **Karmaşıklık Notu:** {complexity_grade}

**Bakım Kolaylığı:**
- **Maintainability Index:** {maintainability_score:.2f}
- **MI Sınıfı:** {test_metrics.get('MI Rank', 'N/A')}
- **Yorum Oranı:** {comment_ratio:.1f}%

**Halstead Metrikleri:**
- **Volume:** {test_metrics.get('Halstead Volume', 'N/A')}
- **Difficulty:** {test_metrics.get('Halstead Difficulty', 'N/A')}
- **Estimated Bugs:** {test_metrics.get('Halstead Bugs', 'N/A')}

**Genel Değerlendirme:** {overall_grade} ({overall_score:.1f}/100) {overall_emoji}
                            """)

st.markdown("---")
st.info("""
💡 **Nasıl Çalışır?**
1. **Boundary Analysis:** Koddan koşulları (if, while) çıkarır ve sınır değerlerini belirler
2. **RL Model:** Coverage'ı maksimize eden inputları keşfeder  
3. **Exception Discovery:** Hata durumlarını test eder
4. **LLM Integration:** OpenAI GPT ile test senaryolarını pytest koduna dönüştürür
""")

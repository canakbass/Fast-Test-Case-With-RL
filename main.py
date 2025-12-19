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

st.markdown("---")
st.info("""
💡 **Nasıl Çalışır?**
1. **Boundary Analysis:** Koddan koşulları (if, while) çıkarır ve sınır değerlerini belirler
2. **RL Model:** Coverage'ı maksimize eden inputları keşfeder  
3. **Exception Discovery:** Hata durumlarını test eder
4. **LLM Integration:** OpenAI GPT ile test senaryolarını pytest koduna dönüştürür
""")

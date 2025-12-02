import streamlit as st
import os
import shutil
import tempfile
from rl_module import train_and_generate_cases
from llm_module import generate_pytest_code
from analiz_module import get_metrics, generate_call_graph
import graphviz

st.set_page_config(page_title="AI Test Generator", layout="wide")

st.title("Reinforcement Learning ile Otomatik TestCase Hazırlama")

# Sidebar
st.sidebar.header("Ayarlar")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

timesteps = st.sidebar.slider("RL Training Timesteps", 500, 10000, 1000)

uploaded_file = st.file_uploader("Python Dosyası Yükle", type="py")

if uploaded_file is not None:
    code_content = uploaded_file.read().decode("utf-8")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Yüklenen Kod")
        st.code(code_content, language="python")

        # Metrics
        if st.button("Kod Analizi Yap"):
            metrics = get_metrics(code_content)
            st.write("### Kod Metrikleri (Radon)")
            st.json(metrics)

            # Graph
            try:
                dot = generate_call_graph(code_content)
                st.write("### Fonksiyon Çağrı Grafiği")
                st.graphviz_chart(dot)
            except Exception as e:
                st.error(f"Grafik oluşturulamadı: {e}")

    with col2:
        st.subheader("Otomatik Test Üretimi")
        if st.button("RL Ajanı ile Test Üret"):
            with st.spinner("RL Ajanı eğitiliyor ve test case'ler aranıyor..."):
                # Save temp file for RL module to import
                # We need to make sure the filename matches what RL module expects if it does imports
                # RL module creates 'temp_module.py' in current directory.

                cases = train_and_generate_cases(code_content, timesteps=timesteps)

                st.success(f"{len(cases)} adet test senaryosu üretildi!")
                st.write("### Üretilen Test Case'ler (Input -> New Coverage)")
                st.dataframe(cases)

                if cases:
                    with st.spinner("LLM ile Pytest koduna çevriliyor..."):
                        pytest_code = generate_pytest_code(cases, code_content)
                        st.write("### Oluşturulan Pytest Kodu")
                        st.code(pytest_code, language="python")

                        st.download_button(
                            label="Test Kodunu İndir",
                            data=pytest_code,
                            file_name="test_generated.py",
                            mime="text/x-python"
                        )

st.markdown("---")
st.info("Not: Bu sistem Stable-Baselines3 (PPO) kullanarak coverage arttıran inputları keşfeder ve OpenAI GPT ile bunları test koduna dönüştürür.")

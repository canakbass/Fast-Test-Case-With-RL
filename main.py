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
        mode = st.radio("Test Case Üretim Modu", ["Hazır Modelle Üret", "Modeli Fine-Tune Et ve Üret"])
        model_path = "ppo_model.zip"
        fine_tune_steps = st.sidebar.slider("Fine-Tune Step", 100, 5000, 500)

        if st.button("Test Case Üret"):
            if mode == "Hazır Modelle Üret":
                with st.spinner("Hazır model yükleniyor ve test case'ler aranıyor..."):
                    cases = train_and_generate_cases(code_content, timesteps=timesteps, mode="load", model_path=model_path)
            else:
                with st.spinner("Model fine-tune ediliyor ve test case'ler aranıyor..."):
                    cases = train_and_generate_cases(code_content, timesteps=timesteps, mode="fine_tune", model_path=model_path, fine_tune_steps=fine_tune_steps)

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

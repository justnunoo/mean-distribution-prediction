# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt

st.set_page_config(page_title="Loss Predictor", layout="wide")

st.title("📉 Loss Distribution Predictor")

uploaded_file = st.file_uploader("Upload Excel file with a 'loss' column", type=["xlsx", "xls"])

if uploaded_file:
    try:
        data = pd.read_excel(uploaded_file)

        if 'loss' not in data.columns:
            st.error("The uploaded file must contain a column named 'loss'.")
        elif not np.issubdtype(data['loss'].dtype, np.number):
            st.error("The 'loss' column must contain numeric values.")
        else:
            loss_data = data['loss'].dropna()
            shape, loc, scale = lognorm.fit(loss_data, floc=0)
            predicted_loss = lognorm.rvs(shape, scale=scale, size=1000)
            mean_loss = np.mean(predicted_loss)

            st.success(f"✅ Predicted Mean Loss: {mean_loss:.2f}")

            # Plot
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(predicted_loss, bins=30, density=True, color='skyblue', alpha=0.7)
            ax.set_title("Predicted Loss Distribution")
            ax.set_xlabel("Loss")
            ax.set_ylabel("Probability")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")

import plotly.express as px
import streamlit as st
from utils import daftar_fitur, hero, muat_data, tata_letak_plotly

hero("Eksplorasi Data (EDA)", "Sebaran fitur sensor berdasarkan kategori aktivitas.")

df = muat_data()
fitur_opsi = daftar_fitur()
default_x = "tBodyAcc-mean()-X" if "tBodyAcc-mean()-X" in fitur_opsi else fitur_opsi[0]
default_y = "tBodyAcc-mean()-Y" if "tBodyAcc-mean()-Y" in fitur_opsi else fitur_opsi[min(1, len(fitur_opsi) - 1)]

col1, col2 = st.columns(2)
with col1:
    fx = st.selectbox("Pilih fitur sumbu X:", fitur_opsi, index=fitur_opsi.index(default_x), key="eda_fx")
with col2:
    fy = st.selectbox("Pilih fitur sumbu Y:", fitur_opsi, index=fitur_opsi.index(default_y), key="eda_fy")

fig_box = px.box(df, x="Activity", y=fx, color="Activity", title=f"Distribusi {fx} per Aktivitas")
st.plotly_chart(tata_letak_plotly(fig_box), width="stretch")

sampel = df.sample(n=min(1500, len(df)), random_state=42)
fig_sc = px.scatter(sampel, x=fx, y=fy, color="Activity", title=f"Scatter {fx} vs {fy} (Sampel 1.500 Baris)")
st.plotly_chart(tata_letak_plotly(fig_sc), width="stretch")

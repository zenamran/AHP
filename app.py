import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
from reportlab.lib import colors
from reportlab.lib.units import cm
from io import BytesIO
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 1. MUST BE FIRST
st.set_page_config(page_title="Vendor Selection Tool", layout="wide")

# Font Registration (Wrapped in try-except for portability)
try:
    pdfmetrics.registerFont(TTFont("DejaVu", "DejaVuSans.ttf"))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", "DejaVuSans-Bold.ttf"))
    FONT_NAME = "DejaVu"
    FONT_BOLD = "DejaVu-Bold"
except:
    FONT_NAME = "Helvetica"
    FONT_BOLD = "Helvetica-Bold"

# ===== THEME SETUP =====
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

st.sidebar.subheader("🎨 Appearance")
st.session_state.dark_mode = st.sidebar.toggle("Dark Mode", value=st.session_state.dark_mode)

# Custom CSS Logic
if st.session_state.dark_mode:
    st.markdown("""
    <style>
    .stApp { background-color: #0F172A; color: #E2E8F0; }
    h1, h2, h3, h4 { color: #F8FAFC; }
    .stSidebar { background-color: #064E3B; }
    .stButton>button { background-color: #16A34A; color: white; border-radius: 12px; }
    .stMetric { background-color: #1E293B; padding: 15px; border-radius: 14px; border-left: 6px solid #16A34A; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp { background-color: #F8FAFC; color: #0F172A; }
    h1, h2, h3, h4 { color: #064E3B; }
    .stSidebar { background-color: #DCFCE7; }
    .stMetric { background-color: white; padding: 15px; border-radius: 14px; border-left: 6px solid #F97316; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Decision Support System (DSS)")
st.subheader("Integrated AHP & Weighted Scoring Model")

# --- 1. SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("⚙️ General Configuration")
    n_suppliers = st.number_input("Number of Suppliers", min_value=2, max_value=10, value=3)
    n_criteria = st.number_input("Number of Criteria", min_value=2, max_value=10, value=3)

# --- 2. NAMES INPUT ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("👥 Supplier Names")
    supplier_names = [st.text_input(f"Supplier {i+1}", f"Supplier {chr(65+i)}", key=f"sup_{i}") for i in range(n_suppliers)]

with col2:
    st.subheader("📋 Criteria Names")
    criteria_names = [st.text_input(f"Criterion {j+1}", f"Criterion {j+1}", key=f"crit_{j}") for i in range(n_criteria)]

# --- 3. AHP MATRIX ---
st.divider()
st.subheader("⚖️ AHP Pairwise Comparison Matrix")
A = np.eye(n_criteria)
for i in range(n_criteria):
    for j in range(i + 1, n_criteria):
        val = st.number_input(f"{criteria_names[i]} vs {criteria_names[j]}", 
                               min_value=0.11, max_value=9.0, value=1.0, step=0.1, key=f"A_{i}_{j}")
        A[i, j] = val
        A[j, i] = 1 / val

# --- 4. PERFORMANCE MATRIX ---
st.divider()
st.subheader("⭐ Performance Scoring (0-10)")
scores_data = np.zeros((n_suppliers, n_criteria))
for i in range(n_suppliers):
    with st.expander(f"Scores for {supplier_names[i]}"):
        cols = st.columns(n_criteria)
        for j in range(n_criteria):
            scores_data[i, j] = cols[j].number_input(f"{criteria_names[j]}", 0.0, 10.0, 5.0, key=f"S_{i}_{j}")

# --- 5. CALCULATIONS ---
eig_vals, eig_vecs = np.linalg.eig(A)
max_eig = np.real(eig_vals.max())
w_ahp = np.real(eig_vecs[:, eig_vals.argmax()])
w_ahp = w_ahp / w_ahp.sum()

RI_table = {1:0, 2:0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}
CI = (max_eig - n_criteria) / (n_criteria - 1) if n_criteria > 1 else 0
CR = CI / RI_table[n_criteria] if n_criteria > 2 else 0

score_ahp_final = np.dot(scores_data, w_ahp)

# --- 6. RESULTS ---
st.divider()
res_col1, res_col2 = st.columns([1, 2])
with res_col1:
    st.subheader("Consistency")
    st.metric("Consistency Ratio (CR)", f"{CR:.2%}")
    if CR < 0.1: st.success("Consistent ✅")
    else: st.error("Inconsistent ❌")

df_ahp = pd.DataFrame({"Supplier": supplier_names, "Score": score_ahp_final}).sort_values(by="Score", ascending=False)
st.subheader("🏆 Final Ranking")
st.dataframe(df_ahp, use_container_width=True)

# --- 7. PDF GENERATION ---
def generate_sonatrach_pv(df_results):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=3.5*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(name="Title", fontName=FONT_BOLD, fontSize=14, alignment=1, spaceAfter=20)
    body_style = ParagraphStyle(name="Normal", fontName=FONT_NAME, fontSize=10, leading=14)

    elements = []
    elements.append(Paragraph("PROCÈS-VERBAL DE LA COMMISSION D’ÉVALUATION DES OFFRES", title_style))
    
    intro_text = f"Analyse multicritère (AHP) réalisée pour la sélection de {n_suppliers} fournisseurs sur la base de {n_criteria} critères."
    elements.append(Paragraph(intro_text, body_style))
    elements.append(Spacer(1, 10))

    # Table
    table_data = [["Rank", "Supplier", "Final Score"]] + \
                 [[i+1, row['Supplier'], f"{row['Score']:.2f}"] for i, row in df_results.iterrows()]
    
    table = Table(table_data, hAlign='CENTER')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#16A34A")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), FONT_BOLD),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(f"Conclusion: Le fournisseur <b>{df_results.iloc[0]['Supplier']}</b> est recommandé.", body_style))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

st.subheader("📄 Export Report")
if st.button("Generate PDF Report"):
    pdf_file = generate_sonatrach_pv(df_ahp)
    st.download_button("📥 Download Official PV", data=pdf_file, file_name="PV_SONATRACH.pdf", mime="application/pdf")

st.caption("Developed by Zennani Amran / Zerguine Moussa.")





























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
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

pdfmetrics.registerFont(TTFont("DejaVu", "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", "DejaVuSans-Bold.ttf"))

# ===== GREEN & ORANGE PROFESSIONAL THEME =====

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

st.sidebar.divider()
st.sidebar.subheader("🎨 Appearance")
st.session_state.dark_mode = st.sidebar.toggle("Dark Mode", value=st.session_state.dark_mode)

if st.session_state.dark_mode:
    st.markdown("""
    <style>
    .stApp { background-color: #0F172A; color: #E2E8F0; }

    h1, h2, h3, h4 { color: #F8FAFC; }

    .stSidebar { background-color: #064E3B; }

    .stButton>button {
        background-color: #16A34A;
        color: white;
        border-radius: 12px;
        font-weight: 600;
        padding: 8px 18px;
    }

    .stButton>button:hover {
        background-color: #F97316;
        color: white;
    }

    .stDataFrame, .stTable {
        background-color: #1E293B;
        border-radius: 12px;
    }

    .stMetric {
        background-color: #1E293B;
        padding: 15px;
        border-radius: 14px;
        border-left: 6px solid #16A34A;
    }

    div[data-testid="stExpander"] {
        background-color: #1E293B;
        border-radius: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <style>
    .stApp { background-color: #F8FAFC; color: #0F172A; }

    h1, h2, h3, h4 { color: #064E3B; }

    .stSidebar { background-color: #DCFCE7; }

    .stButton>button {
        background-color: #16A34A;
        color: white;
        border-radius: 12px;
        font-weight: 600;
        padding: 8px 18px;
    }

    .stButton>button:hover {
        background-color: #F97316;
        color: white;
    }

    .stDataFrame, .stTable {
        background-color: white;
        border-radius: 12px;
    }

    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 14px;
        border-left: 6px solid #F97316;
    }

    div[data-testid="stExpander"] {
        background-color: white;
        border-radius: 12px;
    }
    </style>
    """, unsafe_allow_html=True)



st.set_page_config(page_title="Vendor Selection Tool", layout="wide")

st.title("🚀 Decision Support System (DSS)")
st.subheader("Integrated AHP & Weighted Scoring Model")

# --- 1. SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("⚙️ General Configuration")
    n_suppliers = st.number_input("Number of Suppliers", min_value=2, max_value=10, value=2)
    n_criteria = st.number_input("Number of Criteria", min_value=2, max_value=10, value=2)
    
# --- 2. NAMES INPUT ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("👥 Supplier Names")
    supplier_names = [st.text_input(f"Supplier {i+1}", f"Supplier {chr(65+i)}") for i in range(n_suppliers)]

with col2:
    st.subheader("📋 Criteria Names")
    criteria_names = [st.text_input(f"Criterion {j+1}", f"Criterion {j+1}") for j in range(n_criteria)]

# --- 3. AHP COMPARISON MATRIX ---
st.divider()
st.subheader("⚖️ AHP Pairwise Comparison Matrix (Saaty Scale)")
st.write("Compare the relative importance of criteria (1: Equal, 3: Moderate, 5: Strong, 7: Very Strong, 9: Extreme)")

A = np.eye(n_criteria)
for i in range(n_criteria):
    for j in range(i + 1, n_criteria):
        val = st.number_input(f"How important is {criteria_names[i]} vs {criteria_names[j]}?", 
                               min_value=0.11, max_value=9.0, value=1.0, step=0.1, key=f"A_{i}_{j}")
        A[i, j] = val
        A[j, i] = 1 / val

# --- 4. PERFORMANCE MATRIX (SCORES) ---
st.divider()
st.subheader("⭐ Performance Scoring (Scale 0-10)")
st.write("Enter the raw performance scores for each supplier per criterion.")

scores_data = np.zeros((n_suppliers, n_criteria))
for i in range(n_suppliers):
    with st.expander(f"Scores for {supplier_names[i]}"):
        cols = st.columns(n_criteria)
        for j in range(n_criteria):
            scores_data[i, j] = cols[j].number_input(f"{criteria_names[j]}", 0.0, 10.0, 0.0, key=f"S_{i}_{j}")

# --- 5. MATHEMATICAL CALCULATIONS ---
# AHP Weights & Consistency
eig_vals, eig_vecs = np.linalg.eig(A)
max_eig = np.real(eig_vals.max())
w_ahp = np.real(eig_vecs[:, eig_vals.argmax()])
w_ahp = w_ahp / w_ahp.sum()

# RI Table for Consistency Ratio
RI_table = {1:0, 2:0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}
CI = (max_eig - n_criteria) / (n_criteria - 1)
CR = CI / RI_table[n_criteria] if n_criteria > 2 else 0

# AHP Method Score
score_ahp_final = np.dot(scores_data, w_ahp)
# --- 6. RESULTS & OUTPUT ---
st.divider()
st.header("📊 Final Results & Ranking")

res_col1, res_col2 = st.columns([1, 2])

with res_col1:
    st.subheader("Consistency Check")
    st.metric("Consistency Ratio (CR)", f"{CR:.2%}")
    if CR < 0.1:
        st.success("Consistent Matrix ✅")
    else:
        st.error("Inconsistent Matrix ❌ (Please revise AHP values)")

st.subheader("📋 Criteria Weight AHP Results")

df_scoring = pd.DataFrame({
    "Criteria": criteria_names,
    "Weight": w_ahp
}).sort_values(by="Weight", ascending=False)

st.dataframe(df_scoring, use_container_width=True)
st.subheader("🏆 AHP-Based Weighted Scoring Results")

df_ahp = pd.DataFrame({
    "Supplier": supplier_names,
    "Score": score_ahp_final
}).sort_values(by="Score", ascending=False)

st.dataframe(df_ahp, use_container_width=True)

# --- 7. SENSITIVITY ANALYSIS CHART ---
st.divider()
st.subheader("📈 Sensitivity Analysis")

selected_criterion = st.selectbox(
    "Select the criterion to analyze",
    criteria_names
)

crit_index = criteria_names.index(selected_criterion)

st.write(f"Effect of varying the weight of '{selected_criterion}' on final scores.")

variation = np.linspace(0.05, 0.95, 20)
sens_results = []

for v in variation:
    temp_w = np.copy(w_ahp)
    temp_w[crit_index] = v

    # إعادة توزيع الأوزان على باقي المعايير
    others = [i for i in range(n_criteria) if i != crit_index]
    sum_others = temp_w[others].sum()

    if sum_others > 0:
        temp_w[others] = (temp_w[others] / sum_others) * (1 - v)

    # نحفظ نتيجة كل الموردين عند هذا الوزن
    sens_results.append(np.dot(scores_data, temp_w))

sens_results = np.array(sens_results)  # الشكل يصبح (20, n_suppliers)

# الرسم البياني
fig, ax = plt.subplots(figsize=(10, 5))

for i in range(n_suppliers):
    ax.plot(variation * 100, sens_results[:, i], label=supplier_names[i], linewidth=2)

ax.set_xlabel(f"Weight of {selected_criterion} (%)")
ax.set_ylabel("Score")
ax.set_title("Sensitivity Analysis Graph")
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.grid(True, linestyle='--', alpha=0.7)

st.pyplot(fig)


st.write("---")

st.caption("Developed for Strategic Sourcing and Procurement Analysis.")
st.caption("Developed by Zennani Amran / Zerguine Moussa.")

#.......PV.........
# .......PV.........
#def generate_sonatrach_pv(df_results):
  #  buffer = BytesIO()

    #def header_footer(canvas, doc):
     #   canvas.setFont("Helvetica", 9)
    #    canvas.drawString(2*cm, 28.5*cm, "SONATRACH")
    #    canvas.drawString(2*cm, 28.0*cm, "Direction Approvisionnement")
     #   canvas.drawString(2*cm, 27.5*cm, "Département Achats")

    #    canvas.drawRightString(19*cm, 28.5*cm, "Réf : PV/DSS/2026")
    #    canvas.drawRightString(19*cm, 28.0*cm, "Date : ____ / ____ / 2026")

    #    canvas.line(2*cm, 27.2*cm, 19*cm, 27.2*cm)
    #    canvas.drawRightString(19*cm, 1.2*cm, f"Page {doc.page}")

 #   doc = SimpleDocTemplate(
     #   buffer,
    #    pagesize=A4,
    #    rightMargin=2*cm,
     #   leftMargin=2*cm,
     #   topMargin=3.5*cm,
     #   bottomMargin=2*cm
  #  )

  #  styles = getSampleStyleSheet()

  #  title = ParagraphStyle(
    #   name="Title",
    #   fontName="DejaVu-Bold",
    #   fontSize=16,
     #  spaceAfter=20,
    #   alignment=1 # للتوسيط
 #   )

    # تم تغيير التسمية هنا من normal إلى body لتطابق الاستدعاء في الأسفل
  #  body = ParagraphStyle(
   #   name="Normal",
    #  fontName="DejaVu",
   #   fontSize=11,
    #  leading=14
   # )

  #  elements = []

    # عنوان رسمي
  #  elements.append(Paragraph(
  #      "PROCÈS-VERBAL DE LA COMMISSION D’ÉVALUATION DES OFFRES",
  #      title
  #  ))

  #  elements.append(Spacer(1, 10))

    # نص إداري رسمي
 #   text = """
  #  L’an deux mille vingt-six et le ____ / ____ / 2026, la commission d’évaluation des offres...
  #  L’évaluation a été réalisée selon une méthodologie multicritère (AHP)...
  #  """

  #  for line in text.strip().split("\n"):
  #      if line.strip():
  #          elements.append(Paragraph(line.strip(), body)) # الآن المتغير body معرف ولن يظهر الخطأ

  #  elements.append(Spacer(1, 15))

    # جدول النتائج
#    elements.append(Paragraph("Résultats de l’évaluation :", styles["Heading2"]))

 #   table_data = [list(df_results.columns)] + df_results.values.tolist()

 #   table = Table(table_data, repeatRows=1, hAlign="LEFT")

 #   table.setStyle(TableStyle([
   #     ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
  #      ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
  #      ('FONT', (0,0), (-1,-1), 'Helvetica'),
 #       ('FONTSIZE', (0,0), (-1,-1), 9),
 #       ('ALIGN', (1,1), (-1,-1), 'CENTER'),
    #    ('BOTTOMPADDING', (0,0), (-1,0), 8),
  #  ]))

  #  elements.append(Spacer(1, 10))
  #  elements.append(table)
  # elements.append(Spacer(1, 15))

    # النتيجة النهائية
  #  best = df_results.iloc[0]["Supplier"]

 #   conclusion = f"""
  #  Au vu des résultats obtenus, la commission d’évaluation propose de retenir l’offre du fournisseur
  #  <b>{best}</b>, ayant obtenu la meilleure note globale conformément aux critères définis.
  #  """

  #  elements.append(Paragraph(conclusion, body))
   # elements.append(Spacer(1, 20))

    # توقيعات رسمية
  #  elements.append(Paragraph("Fait pour servir et valoir ce que de droit.", body))
  #  elements.append(Spacer(1, 30))

  #  signature_table = Table([
  #      ["Président de la commission", "Membre", "Membre"],
  #      ["Nom : ____________________", "Nom : ____________________", "Nom : ____________________"],
  #      ["Signature : ______________", "Signature : _____________", "Signature : _____________"],
  #  ], colWidths=[6*cm, 6*cm, 6*cm])

  #  signature_table.setStyle(TableStyle([
 #       ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
 #       ('ALIGN', (0,0), (-1,-1), 'CENTER'),
#        ('FONTSIZE', (0,0), (-1,-1), 9),
#    ]))

  #  elements.append(signature_table)

 #   doc.build(elements, onFirstPage=header_footer, onLaterPages=header_footer)

 #   buffer.seek(0)
#    return buffer

#st.subheader("📄 Procès-Verbal")

#pdf = generate_sonatrach_pv(df_ahp)

#st.download_button(
#    "📥 Télécharger le PV",
#    data=pdf,
#    file_name="PV_Evaluation_SONATRACH.pdf",
#    mime="application/pdf"
#)




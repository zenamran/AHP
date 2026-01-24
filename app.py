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

# --- 1. إعداد الصفحة (يجب أن يكون أول أمر Streamlit) ---
st.set_page_config(page_title="Vendor Selection Tool", layout="wide", page_icon="🚀")

# تسجيل الخطوط (تأكد من وجود الملفات في مجلد المشروع)
try:
    pdfmetrics.registerFont(TTFont("DejaVu", "DejaVuSans.ttf"))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", "DejaVuSans-Bold.ttf"))
    FONT_NAME, FONT_BOLD = "DejaVu", "DejaVu-Bold"
except:
    FONT_NAME, FONT_BOLD = "Helvetica", "Helvetica-Bold"

# --- 2. إعدادات الثيمات (Theme Engine) ---
if "theme" not in st.session_state:
    st.session_state.theme = "Green & Orange Pro"

with st.sidebar:
    st.subheader("🎨 المظهر")
    theme_choice = st.radio("اختر نمط العرض:", ["Green & Orange Pro", "Dark Mode", "Light Mode"], horizontal=True)
    st.session_state.theme = theme_choice

# تطبيق الـ CSS المخصص بناءً على الاختيار
if st.session_state.theme == "Green & Orange Pro":
    st.markdown("""
    <style>
    .stApp { background-color: #F8FAF5; color: #1B2E1B; }
    h1, h2, h3 { color: #1A531B !important; font-weight: 800; }
    .stSidebar { background-color: #E8F5E9; border-right: 2px solid #1A531B; }
    .stButton>button { 
        background-color: #F97316; color: white; border-radius: 8px; 
        border: none; font-weight: bold; transition: 0.3s;
    }
    .stButton>button:hover { background-color: #1A531B; transform: scale(1.02); }
    .stMetric { 
        background-color: white; padding: 15px; border-radius: 12px; 
        border-left: 6px solid #F97316; box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    div[data-testid="stExpander"] { background-color: white; border: 1px solid #C8E6C9; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)
elif st.session_state.theme == "Dark Mode":
    st.markdown("<style>.stApp { background-color: #0F172A; color: white; }</style>", unsafe_allow_html=True)

# --- 3. واجهة المستخدم المدخلات ---
st.title("🚀 نظام دعم القرار (DSS)")
st.subheader("منهجية AHP للتقييم والمفاضلة")

col_cfg1, col_cfg2 = st.sidebar.columns(2)
n_suppliers = col_cfg1.number_input("الموردين", 2, 10, 3)
n_criteria = col_cfg2.number_input("المعايير", 2, 10, 3)

st.divider()
c1, c2 = st.columns(2)
with c1:
    st.markdown("### 👥 أسماء الموردين")
    supplier_names = [st.text_input(f"المورد {i+1}", f"Supplier {chr(65+i)}", key=f"s{i}") for i in range(n_suppliers)]
with c2:
    st.markdown("### 📋 المعايير")
    criteria_names = [st.text_input(f"المعيار {j+1}", f"Criterion {j+1}", key=f"c{j}") for j in range(n_criteria)]

# --- 4. العمليات الحسابية (AHP) ---
st.divider()
st.subheader("⚖️ مصفوفة المقارنة الزوجية (Saaty Scale)")
A = np.eye(n_criteria)
for i in range(n_criteria):
    for j in range(i + 1, n_criteria):
        val = st.number_input(f"أهمية {criteria_names[i]} مقارنة بـ {criteria_names[j]}", 0.1, 9.0, 1.0, key=f"A{i}{j}")
        A[i, j] = val
        A[j, i] = 1 / val

# حساب الأوزان
eig_vals, eig_vecs = np.linalg.eig(A)
w_ahp = np.real(eig_vecs[:, eig_vals.argmax()])
w_ahp /= w_ahp.sum()

# حساب نسبة الاتساق (CR)
max_eig = np.real(eig_vals.max())
CI = (max_eig - n_criteria) / (n_criteria - 1) if n_criteria > 1 else 0
RI = {1:0, 2:0, 3:0.58, 4:0.9, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}
CR = CI / RI[n_criteria] if n_criteria > 2 else 0

# --- 5. تقييم الموردين ---
st.divider()
st.subheader("⭐ درجات الأداء (0-10)")
scores_data = np.zeros((n_suppliers, n_criteria))
for i in range(n_suppliers):
    with st.expander(f"تقييم {supplier_names[i]}"):
        cols = st.columns(n_criteria)
        for j in range(n_criteria):
            scores_data[i, j] = cols[j].number_input(f"{criteria_names[j]}", 0.0, 10.0, 5.0, key=f"sc{i}{j}")

final_scores = np.dot(scores_data, w_ahp)
df_results = pd.DataFrame({"Supplier": supplier_names, "Score": final_scores}).sort_values("Score", ascending=False)

# --- 6. عرض النتائج ---
st.divider()
st.header("📊 النتائج النهائية")
res_c1, res_c2 = st.columns([1, 2])
with res_c1:
    st.metric("نسبة الاتساق (CR)", f"{CR:.2%}")
    if CR < 0.1: st.success("المصفوفة متسقة ✅")
    else: st.error("المصفوفة غير متسقة ❌")

with res_c2:
    st.dataframe(df_results, use_container_width=True)

# --- 7. وظيفة توليد PDF (تم إصلاح NameError) ---
def generate_pdf(df):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # تعريف التنسيقات
    title_style = ParagraphStyle("Title", fontName=FONT_BOLD, fontSize=16, alignment=1, spaceAfter=20)
    body_style = ParagraphStyle("Body", fontName=FONT_NAME, fontSize=11, leading=14) # تم تغيير الاسم هنا
    
    elements = []
    elements.append(Paragraph("PROCÈS-VERBAL D’ÉVALUATION", title_style))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"بناءً على تحليل AHP، تم ترتيب الموردين كما يلي:", body_style))
    
    # الجدول
    data = [["Rank", "Supplier", "Score"]] + [[i+1, r[0], f"{r[1]:.2f}"] for i, r in enumerate(df.values)]
    table = Table(data, colWidths=[2*cm, 7*cm, 4*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A531B")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,-1), FONT_NAME),
        ('ALIGN', (0,0), (-1,-1), 'CENTER')
    ]))
    elements.append(table)
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

st.divider()
if st.button("📥 تحميل محضر الاجتماع (PV)"):
    pdf = generate_pdf(df_results)
    st.download_button("تأكيد التحميل", data=pdf, file_name="PV_Evaluation.pdf", mime="application/pdf")

st.caption("Developed by Zennani Amran / Zerguine Moussa.")



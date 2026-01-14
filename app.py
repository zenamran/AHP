import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Vendor Selection Tool", layout="wide")

st.title("🚀 Decision Support System (DSS)")
st.subheader("Integrated AHP & Weighted Scoring Model")

# --- 1. SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("⚙️ General Configuration")
    n_suppliers = st.number_input("Number of Suppliers", min_value=2, max_value=10, value=5)
    n_criteria = st.number_input("Number of Criteria", min_value=2, max_value=10, value=7)
    
    st.divider()
    st.header("⚖️ Method Weighting")
    w_scoring_ratio = st.slider("Scoring Method Weight", 0.0, 1.0, 0.5)
    w_ahp_ratio = 1.0 - w_scoring_ratio
    st.info(f"Final Score = ({w_scoring_ratio} × Scoring) + ({w_ahp_ratio:.1f} × AHP)")

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
            scores_data[i, j] = cols[j].number_input(f"{criteria_names[j]}", 0.0, 10.0, 7.0, key=f"S_{i}_{j}")

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

# Scoring Method (Using Uniform Weights for the criteria)
weights_scoring = np.ones(n_criteria) / n_criteria
score_scoring = np.dot(scores_data, weights_scoring)

# AHP Method Score
score_ahp_final = np.dot(scores_data, w_ahp)

# Hybrid Final Score
final_score = (score_scoring * w_scoring_ratio) + (score_ahp_final * w_ahp_ratio)

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

with res_col2:
    st.subheader("Comparison Table")
    df_results = pd.DataFrame({
        "Supplier": supplier_names,
        "Scoring Score": score_scoring,
        "AHP Score": score_ahp_final,
        "Combined Final Score": final_score
    }).sort_values(by="Combined Final Score", ascending=False)
    st.dataframe(df_results.style.highlight_max(axis=0, color='lightgreen'))

# --- 7. SENSITIVITY ANALYSIS CHART ---
st.divider()
st.subheader("📈 Sensitivity Analysis (Criterion 1)")
st.write(f"Effect of varying the weight of '{criteria_names[0]}' on final scores.")

variation = np.linspace(0.05, 0.95, 20)
sens_results = []

for v in variation:
    # Adjust weights proportionally
    temp_w = np.copy(w_ahp)
    temp_w[0] = v
    # Normalize other weights
    sum_others = temp_w[1:].sum()
    if sum_others > 0:
        temp_w[1:] = (temp_w[1:] / sum_others) * (1 - v)
    sens_results.append(np.dot(scores_data, temp_w))

fig, ax = plt.subplots(figsize=(10, 5))
for i in range(n_suppliers):
    ax.plot(variation * 100, [s[i] for s in sens_results], label=supplier_names[i], linewidth=2)

ax.set_xlabel(f"Weight of {criteria_names[0]} (%)")
ax.set_ylabel("Total Score")
ax.set_title("Sensitivity Analysis Graph")
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.grid(True, linestyle='--', alpha=0.7)
st.pyplot(fig)

st.write("---")
st.caption("Developed for Strategic Sourcing and Procurement Analysis.")
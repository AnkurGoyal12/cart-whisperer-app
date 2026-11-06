# Streamlit app, stage 1: page setup, model dropdown and test CSV upload
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
st.set_page_config(page_title="Cart Whisperer", layout="wide")
# Warm amber palette applied through custom CSS
st.markdown(
    """
    <style>
    .stApp { background-color: #fff8f0; color: #3b2a1a; }
    h1, h2, h3 { color: #7a3e00; }
    [data-testid="stSidebar"] { background-color: #ffe9cf; }
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border-left: 6px solid #e07b00;
        padding: 10px 14px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_VAULT = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}


@st.cache_resource
def unseal_model(vault_path):
    return joblib.load(vault_path)


st.title("Cart Whisperer: will this visit end in a purchase?")
st.write("Choose a trained model, drop in a test CSV, and see how well it spots the buyers.")

chosen_algo = st.sidebar.selectbox("Pick a model", list(MODEL_VAULT.keys()))
csv_dropzone = st.sidebar.file_uploader("Drop your test CSV here", type="csv")

if csv_dropzone is None:
    st.info("Waiting for a test CSV in the sidebar.")
    st.stop()

incoming_sessions = pd.read_csv(csv_dropzone)
st.write(f"Sessions loaded: {incoming_sessions.shape[0]}, columns: {incoming_sessions.shape[1]}")
st.dataframe(incoming_sessions.head())

# Stage 2: validate the upload, run the chosen saved model and show the six metrics
model_unit = unseal_model(MODEL_VAULT[chosen_algo])
needed_inputs = list(model_unit.feature_names_in_)
absent_cols = [col for col in needed_inputs + ["Revenue"] if col not in incoming_sessions.columns]
if absent_cols:
    st.error(f"The uploaded file is missing these columns: {absent_cols}")
    st.stop()

true_flags = incoming_sessions["Revenue"].astype(int)
pred_flags = model_unit.predict(incoming_sessions[needed_inputs])
pred_chance = model_unit.predict_proba(incoming_sessions[needed_inputs])[:, 1]

st.subheader(f"Scores for {chosen_algo}")
score_tiles = st.columns(6)
score_tiles[0].metric("Accuracy", f"{accuracy_score(true_flags, pred_flags):.4f}")
score_tiles[1].metric("AUC", f"{roc_auc_score(true_flags, pred_chance):.4f}")
score_tiles[2].metric("Precision", f"{precision_score(true_flags, pred_flags, zero_division=0):.4f}")
score_tiles[3].metric("Recall", f"{recall_score(true_flags, pred_flags):.4f}")
score_tiles[4].metric("F1", f"{f1_score(true_flags, pred_flags):.4f}")
score_tiles[5].metric("MCC", f"{matthews_corrcoef(true_flags, pred_flags):.4f}")

# Plain-English verdict on how many real buyers the model found
found_buyers = int(((true_flags == 1) & (pred_flags == 1)).sum())
all_buyers = int((true_flags == 1).sum())
st.success(f"{chosen_algo} caught {found_buyers} of the {all_buyers} real buyers in this file.")

# Stage 3: draw the confusion matrix and show the classification report side by side
st.subheader("Where the model gets it right and wrong")
grid_pane, report_pane = st.columns(2)

with grid_pane:
    grid_figure, grid_axes = plt.subplots(figsize=(4, 3.5))
    ConfusionMatrixDisplay.from_predictions(
        true_flags,
        pred_flags,
        display_labels=["No purchase", "Purchase"],
        cmap="Purples",
        ax=grid_axes,
        colorbar=False,
    )
    st.pyplot(grid_figure)

with report_pane:
    report_frame = pd.DataFrame(
        classification_report(
            true_flags,
            pred_flags,
            target_names=["No purchase", "Purchase"],
            output_dict=True,
            zero_division=0,
        )
    ).transpose().round(3)
    st.dataframe(report_frame)

# Extra table: the ten sessions the chosen model rates as most likely to buy
st.subheader("Ten visits the model rates most likely to buy")
ranked_view = incoming_sessions.assign(Buy_Chance=pred_chance.round(3), Model_Call=pred_flags)
ranked_view = ranked_view[["Buy_Chance", "Model_Call"] + list(incoming_sessions.columns)]
st.dataframe(ranked_view.sort_values("Buy_Chance", ascending=False).head(10))
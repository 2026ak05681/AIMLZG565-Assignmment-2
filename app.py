import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Obesity Level"
)

# from the training 
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib"
}

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("model/preprocessor.joblib")
    feature_names = joblib.load("model/feature_names.joblib")
    class_labels = joblib.load("model/class_labels.joblib")
    loaded_models = {}
    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            loaded_models[name] = joblib.load(path)
    return preprocessor, feature_names, class_labels, loaded_models

def compute_metrics(model, X_proc, y_true):
    y_pred = model.predict(X_proc)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_proc)
    else:
        y_prob = None

    auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted") if y_prob is not None else 0.0

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "y_pred": y_pred,
        "y_prob": y_prob
    }

def main():
    st.title("Obesity Level Prediction")
    st.caption("AIMLZG565-Assignmment-2-2026ak05681")

    try:
        preprocessor, feature_names, class_labels, models = load_artifacts()
    except Exception as e:
        st.error(f"Error loading saved models {e}")
        st.stop()

    with st.sidebar:
        st.header("Test Dataset Input")
        uploaded_file = st.file_uploader(
            "upload test data csv",
            type=["csv"]
        )
        use_default = st.checkbox("use committed test_data.csv", value=(uploaded_file is None))

        st.divider()
        st.header("Model Selection")
        selected_model_name = st.selectbox(
            "choose a pre trained classifier model:",
            options=list(models.keys())
        )

        if os.path.exists("test_data.csv"):
            with open("test_data.csv", "rb") as f:
                st.download_button(
                    label="repo test_data.csv",
                    data=f,
                    file_name="test_data.csv",
                    mime="text/csv"
                )

    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded uploaded file: `{uploaded_file.name}` ({df.shape[0]} rows, {df.shape[1]} columns)")
    elif use_default and os.path.exists("test_data.csv"):
        df = pd.read_csv("test_data.csv")
        st.info(f"using repo `test_data.csv` ({df.shape[0]} rows, {df.shape[1]} columns)")
    else:
        st.warning("Error: upload a test csv file with correct format.")
        st.stop()

    with st.expander("View Test Dataset", expanded=False):
        st.dataframe(df.head(), use_container_width=True)

    target_col = "NObeyesdad" if "NObeyesdad" in df.columns else ("target" if "target" in df.columns else None)
    missing_cols = [c for c in feature_names if c not in df.columns]

    if missing_cols or target_col is None:
        st.error(
            f"Error: Uploaded CSV must contain all {len(feature_names)} features and the `NObeyesdad` target column. "
            f"Missing features: {missing_cols[:5]}"
        )
        st.stop()

    X_test = df[feature_names]
    y_test = df[target_col]
    X_test_proc = preprocessor.transform(X_test)

    all_metrics = {}
    summary_rows = []
    for m_name, m_obj in models.items():
        res = compute_metrics(m_obj, X_test_proc, y_test)
        all_metrics[m_name] = res
        summary_rows.append({
            "Model Name": m_name,
            "Accuracy": round(res["Accuracy"], 4),
            "AUC": round(res["AUC"], 4),
            "Precision": round(res["Precision"], 4),
            "Recall": round(res["Recall"], 4),
            "F1": round(res["F1"], 4),
            "MCC": round(res["MCC"], 4)
        })

    st.subheader(f"Test Results: {selected_model_name}")
    sel_res = all_metrics[selected_model_name]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{sel_res['Accuracy']:.4f}")
    c2.metric("AUC Score", f"{sel_res['AUC']:.4f}")
    c3.metric("Precision", f"{sel_res['Precision']:.4f}")
    c4.metric("Recall", f"{sel_res['Recall']:.4f}")
    c5.metric("F1 Score", f"{sel_res['F1']:.4f}")
    c6.metric("MCC Score", f"{sel_res['MCC']:.4f}")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("Confusion Matrix")
        labels = sorted(list(set(y_test)))
        cm = confusion_matrix(y_test, sel_res["y_pred"], labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=[lbl.replace("_", "\n") for lbl in labels],
            yticklabels=[lbl.replace("_", "\n") for lbl in labels],
            ax=ax
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_title(f"Confusion Matrix — {selected_model_name}")
        st.pyplot(fig)

    with col_right:
        st.markdown("Classification Report")
        report_dict = classification_report(
            y_test, sel_res["y_pred"],
            output_dict=True,
            zero_division=0
        )
        report_df = pd.DataFrame(report_dict).transpose().round(4)
        st.dataframe(report_df, use_container_width=True)

        st.markdown("Target Class Distribution in Test Set")
        class_dist = y_test.value_counts()
        st.bar_chart(class_dist)

    st.divider()

    st.subheader("All 5 pretrained model comparison")
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(
        summary_df.style.highlight_max(
            subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
        ),
        use_container_width=True
    )

if __name__ == "__main__":
    main()

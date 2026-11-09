import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)
from sklearn.calibration import CalibratedClassifierCV

RANDOM_SEED = 42

def main():

    #If not existing then create model folder
    os.makedirs("model", exist_ok=True)

    # load UCI Obesity Levels Dataset
    # instances: 2,111 
    # eatures: 16 input features 
    csv_path = os.path.join("data", "ObesityDataSet.csv")
    df = pd.read_csv(csv_path)

    print(f"data set success: {df.shape[0]} rows, {df.shape[1]-1} features")

    # output: NObeyesdad, 7 obesity levels
    target_col = "NObeyesdad"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # use stratify split for more correct split for classification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    # Save test_data.csv
    test_df = X_test.copy()
    test_df[target_col] = y_test.values
    test_df.to_csv("test_data.csv", index=False)
    print(f"done with test_data.csv ({test_df.shape[0]} rows, {test_df.shape[1]} columns).")

    # scale and encode the numeric and categories
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # save these belwo because need for ui streamlit app
    joblib.dump(preprocessor, "model/preprocessor.joblib")
    joblib.dump(list(X.columns), "model/feature_names.joblib")
    joblib.dump(sorted(y.unique().tolist()), "model/class_labels.joblib")

    # create assignemnt required classification models
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
            "logistic_regression.joblib"
        ),
        "Decision Tree": (
            DecisionTreeClassifier(max_depth=10, random_state=RANDOM_SEED),
            "decision_tree.joblib"
        ),
        "KNN": (
            KNeighborsClassifier(n_neighbors=5, weights="distance"),
            "knn.joblib"
        ),
        "Naive Bayes": (
            GaussianNB(),
            "naive_bayes.joblib"
        ),
        # used below to fix warning
        # Warning FutureWarning: The `probability` parameter was deprecated in 1.9 and will be removed in version 1.11. Use `CalibratedClassifierCV(SVC(), ensemble=False)` instead of `SVC(probability=True)`
                "Support Vector Machine": (
            CalibratedClassifierCV(
                SVC(kernel="rbf", C=10.0, random_state=RANDOM_SEED),
                ensemble=False
            ),
            "svm.joblib"
        )
    }

    results = []

    # trainmodels
    print("training models")
    for model_name, (clf, filename) in models.items():
        clf.fit(X_train_proc, y_train)
        joblib.dump(clf, os.path.join("model", filename))

        y_pred = clf.predict(X_test_proc)
        y_prob = clf.predict_proba(X_test_proc)

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        results.append({
            "Model": model_name,
            "Accuracy": round(acc, 4),
            "AUC": round(auc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "MCC": round(mcc, 4)
        })

    results_df = pd.DataFrame(results)

    print("scores for trained models")
    print(results_df.to_string())

if __name__ == "__main__":
    main()

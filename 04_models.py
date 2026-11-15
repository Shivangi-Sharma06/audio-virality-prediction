import json, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import common


def main():
    with open("results/selected_features.json") as f:
        features = json.load(f)["selected_features"]
    with open("results/pca_info.json") as f:
        n_comp = json.load(f)["n_components_chosen"]

    X_train, X_test, y_train, y_test = common.get_train_test()
    X_tr, X_te = X_train[features], X_test[features]

    # Models & setups: RF is scale-invariant without PCA; PCA requires StandardScaler
    configs = [
        ("LogisticRegression", "without_pca", Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])),
        ("LogisticRegression", "with_pca", Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_comp, random_state=42)), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])),
        ("RandomForest", "without_pca", Pipeline([("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1))])),
        ("RandomForest", "with_pca", Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_comp, random_state=42)), ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1))]))
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results, reports, roc_data = [], [], []

    for name, setup, pipe in configs:
        cv_s = cross_validate(pipe, X_tr, y_train, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=-1)
        t0 = time.time(); pipe.fit(X_tr, y_train); fit_sec = time.time() - t0
        y_pred, y_prob = pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1]

        acc, f1, auc = accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average="macro"), roc_auc_score(y_test, y_prob)
        results.append({
            "Model": name, "Setup": setup,
            "CV_Acc": f"{cv_s['test_accuracy'].mean():.4f} +/- {cv_s['test_accuracy'].std():.4f}",
            "CV_Macro_F1": f"{cv_s['test_f1_macro'].mean():.4f} +/- {cv_s['test_f1_macro'].std():.4f}",
            "Test_Acc": round(acc, 4), "Test_Prec": round(precision_score(y_test, y_pred, average="macro"), 4),
            "Test_Rec": round(recall_score(y_test, y_pred, average="macro"), 4),
            "Test_Macro_F1": round(f1, 4), "Test_ROC_AUC": round(auc, 4), "Train_Time_s": round(fit_sec, 3)
        })

        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(cmap="Blues")
        plt.title(f"CM: {name} ({setup})"); plt.tight_layout()
        plt.savefig(f"results/cm_{name}_{setup}.png", dpi=200); plt.close()
        reports.append(f"=== {name} ({setup}) ===\n" + classification_report(y_test, y_pred, digits=4))
        roc_data.append((name, setup, auc, *roc_curve(y_test, y_prob)[:2]))

    res_df = pd.DataFrame(results)
    print("=== TASK IV: MODEL EVALUATION COMPARISON ===")
    print(res_df.to_string(index=False))
    best = res_df.loc[res_df["Test_Macro_F1"].idxmax()]
    print(f"\nBest Model: {best['Model']} ({best['Setup']}) with Test Macro F1 = {best['Test_Macro_F1']}")
    res_df.to_csv("results/model_results.csv", index=False)
    with open("results/classification_reports.txt", "w") as f:
        f.write("\n\n".join(reports))

    plt.figure(figsize=(7, 5))
    for name, setup, auc, fpr, tpr in roc_data:
        plt.plot(fpr, tpr, label=f"{name} [{setup}] (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance (AUC = 0.500)")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison"); plt.legend(); plt.tight_layout()
    plt.savefig("results/roc_curves.png", dpi=300); plt.close()


if __name__ == "__main__":
    main()

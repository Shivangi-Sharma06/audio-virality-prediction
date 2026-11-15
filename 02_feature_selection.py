import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import common


def main():
    X_train, _, y_train, _ = common.get_train_test()

    # Step 1: Correlation filter to eliminate redundant features (|r| > 0.85)
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped_corr = [c for c in upper.columns if any(upper[c] > 0.85)]
    print("=== TASK II: FEATURE SELECTION (ON TRAIN SET) ===")
    print(f"Features dropped by correlation (|r| > 0.85): {dropped_corr if dropped_corr else 'None'}")
    X_f = X_train.drop(columns=dropped_corr)

    # Step 2: Mutual Information filter (captures non-linear dependencies with target)
    mi_scores = mutual_info_classif(X_f, y_train, random_state=42)

    scores = pd.DataFrame({
        "feature": X_f.columns,
        "mi_score": mi_scores
    }).sort_values("mi_score", ascending=False).reset_index(drop=True)

    # Retain top half of candidate features (at least 5)
    n_keep = max(5, int(np.ceil(len(scores) / 2.0)))
    scores["selected"] = False
    scores.loc[:n_keep - 1, "selected"] = True

    print("\nMutual Information Feature Ranking (Top Selected):")
    print(scores.to_string(index=False))

    # Save scores and selection metadata
    scores.to_csv("results/feature_scores.csv", index=False)
    selected_features = scores.loc[scores["selected"], "feature"].tolist()
    with open("results/selected_features.json", "w") as f:
        json.dump({"dropped_by_correlation": dropped_corr, "selected_features": selected_features}, f, indent=2)

    # Horizontal bar plot of Mutual Information scores
    plt.figure(figsize=(8, 6))
    plot_df = scores.sort_values("mi_score", ascending=True)
    colors = ["#2b5c8f" if s else "#c0c0c0" for s in plot_df["selected"]]
    plt.barh(plot_df["feature"], plot_df["mi_score"], color=colors)
    plt.xlabel("Mutual Information Score")
    plt.title("Mutual Information Feature Importance (Blue = Selected Top Half)")
    plt.tight_layout()
    plt.savefig("results/feature_importance.png", dpi=300)
    plt.close()
    print("\nSaved results/feature_scores.csv, selected_features.json, feature_importance.png")


if __name__ == "__main__":
    main()

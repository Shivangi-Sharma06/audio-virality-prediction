import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import common


def main():
    with open("results/selected_features.json") as f:
        selected_features = json.load(f)["selected_features"]

    X_train, _, y_train, _ = common.get_train_test()
    # Standardize features so features with larger units don't dominate variance
    X_scaled = StandardScaler().fit_transform(X_train[selected_features])
    pca = PCA(random_state=42).fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    exp_var, cum_var = pca.explained_variance_ratio_, np.cumsum(pca.explained_variance_ratio_)
    print("=== TASK III: PRINCIPAL COMPONENT ANALYSIS (PCA) ===")
    for i, (ev, cv) in enumerate(zip(exp_var, cum_var), 1):
        print(f"  PC{i:02d}: Explained = {ev*100:5.2f}% | Cumulative = {cv*100:5.2f}%")

    n_comp = int(np.argmax(cum_var >= 0.95) + 1)
    var_retained = float(cum_var[n_comp - 1])
    print(f"\nChosen Components (>=95% variance): {n_comp} ({var_retained*100:.2f}% retained)")

    # 1. Scree plot with cumulative variance line
    plt.figure(figsize=(7, 4.5))
    x_axis = range(1, len(exp_var) + 1)
    plt.bar(x_axis, exp_var * 100, alpha=0.6, label="Individual Variance", color="#4C72B0")
    plt.plot(x_axis, cum_var * 100, marker="o", color="#C44E52", label="Cumulative Variance")
    plt.axhline(95, color="grey", linestyle="--", label="95% Threshold")
    plt.axvline(n_comp, color="green", linestyle=":", label=f"Selected Cutoff ({n_comp} PCs)")
    plt.xlabel("Principal Component"); plt.ylabel("Explained Variance (%)")
    plt.title("PCA Scree & Cumulative Variance Plot"); plt.legend(); plt.tight_layout()
    plt.savefig("results/pca_scree.png", dpi=300); plt.close()

    # 2. PC1 vs PC2 scatter plot (sample up to 3000 points for clear visualization)
    sample_idx = np.random.RandomState(42).choice(len(X_train), size=min(3000, len(X_train)), replace=False)
    plt.figure(figsize=(6.5, 5))
    scatter = plt.scatter(X_pca[sample_idx, 0], X_pca[sample_idx, 1], c=y_train.iloc[sample_idx], cmap="coolwarm", alpha=0.5)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PC1 vs PC2 Projection (3,000 Train Samples)")
    cbar = plt.colorbar(scatter, ticks=[0, 1])
    cbar.ax.set_yticklabels(["Non-Viral (0)", "Viral (1)"])
    plt.tight_layout(); plt.savefig("results/pca_scatter.png", dpi=300); plt.close()

    # 3. Loadings of first 3 components
    loadings = pd.DataFrame(pca.components_[:3].T, index=selected_features, columns=["PC1", "PC2", "PC3"])
    loadings.to_csv("results/pca_loadings.csv")
    with open("results/pca_info.json", "w") as f:
        json.dump({"n_features_before": len(selected_features), "n_components_chosen": n_comp,
                   "variance_retained": round(var_retained, 4)}, f, indent=2)
    print("Saved results/pca_scree.png, pca_scatter.png, pca_loadings.csv, pca_info.json")


if __name__ == "__main__":
    main()

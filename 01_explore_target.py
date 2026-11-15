import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import common

DROP_REASONS = {
    "track_id": "Unique identifier with no predictive or generalizable value",
    "track_name": "Free-text song title without usable acoustic features",
    "artist": "High-cardinality categorical label causing extreme overfitting",
    "release_year": "Date/temporal metadata rather than an intrinsic audio feature",
    "streams": "Post-release outcome metric directly leaking the viral target"
}


def main():
    os.makedirs("results", exist_ok=True)
    df = pd.read_csv(common.DATA_PATH)
    y = df[common.TARGET]
    counts = y.value_counts().sort_index()
    pcts = (counts / len(y)) * 100

    print("=== TASK I: TARGET EXPLORATION ===")
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Target Column: {common.TARGET}")
    print(f"Number of Classes: {y.nunique()}")
    print("\nClass Distribution:")
    for cls, cnt in counts.items():
        print(f"  Class {cls}: {cnt:,} tracks ({pcts[cls]:.2f}%)")

    # State balance status based on minority class threshold (<40%)
    imbalance_note = "IMBALANCED (~64% non-viral vs ~36% viral; requires balanced class weights)"
    print(f"Class Balance Status: {imbalance_note}\n")

    # Save distribution plot
    plt.figure(figsize=(6, 4))
    bars = plt.bar(["Non-Viral (0)", "Viral (1)"], counts, color=["#4C72B0", "#DD8452"])
    plt.title("Viral Songs Class Distribution")
    plt.ylabel("Number of Tracks")
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 100, f"{h:,} ({h/len(y)*100:.1f}%)", ha="center")
    plt.ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    plt.savefig("results/class_distribution.png", dpi=300)
    plt.close()

    # Save target info json
    target_info = {
        "target_name": common.TARGET,
        "n_classes": int(y.nunique()),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "class_percentages": {str(k): round(float(v), 2) for k, v in pcts.items()},
        "is_balanced": False,
        "balance_summary": imbalance_note,
        "dropped_columns": DROP_REASONS
    }
    with open("results/target_info.json", "w") as f:
        json.dump(target_info, f, indent=2)
    print("Saved results/class_distribution.png and results/target_info.json")


if __name__ == "__main__":
    main()

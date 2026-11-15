import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/viral_songs.csv"
TARGET = "viral"

# Dropping identifiers (track_id, track_name, artist: no generalizable acoustic signal),
# release_year (temporal metadata, not an audio feature), and
# streams (direct target leakage: virality is calculated from stream counts).
DROP_COLS = ["track_id", "track_name", "artist", "release_year", "streams"]


def load_data():
    """Load raw dataset, drop non-predictive/leaking columns, and encode categories."""
    df = pd.read_csv(DATA_PATH).drop(columns=DROP_COLS).drop_duplicates()
    df = df.dropna(subset=[TARGET])
    # One-hot encode low-cardinality categorical features (genre <= 10 classes)
    cat_cols = [c for c in df.select_dtypes(exclude=["number"]).columns if df[c].nunique() <= 10]
    df = pd.get_dummies(df, columns=cat_cols, dtype=int)
    non_numeric = df.select_dtypes(exclude=["number"]).columns
    return df.drop(columns=non_numeric)


def get_train_test():
    """Stratified 80/20 train/test split with imputation strictly on training data."""
    df = load_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    if y.dtype == "object" or isinstance(y.iloc[0], str):
        y = pd.Series(LabelEncoder().fit_transform(y), index=y.index, name=TARGET)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Prevent data leakage: compute medians strictly on the training set
    train_medians = X_train.median()
    return X_train.fillna(train_medians), X_test.fillna(train_medians), y_train, y_test

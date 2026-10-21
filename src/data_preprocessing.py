from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "priceUSD"
NUM_COLS = ["car_age", "mileage_km", "mileage_per_year",
            "engine_volume_liters", "is_high_mileage"]
CAT_COLS = ["make", "model", "condition", "fuel_type", "color",
            "transmission", "drive_unit", "segment"]

def get_preprocessor():
    num = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore",
                                             min_frequency=20,
                                             sparse_output=False))])
    return ColumnTransformer([("num", num, NUM_COLS),
                              ("cat", cat, CAT_COLS)])

def get_split(df):
    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42)
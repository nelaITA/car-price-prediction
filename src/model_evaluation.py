import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data_cleaning import load_data, clean_data
from src.feature_engineering import add_features
from src.data_preprocessing import get_split

def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    return {"MAE": mean_absolute_error(y_test, pred),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R2": r2_score(y_test, pred)}

if __name__ == "__main__":
    df = add_features(clean_data(load_data()))
    _, X_test, _, y_test = get_split(df)
    model = joblib.load("models/car_price_model.joblib")
    m = evaluate(model, X_test, y_test)
    print({k: round(v, 3) for k, v in m.items()})
    print(f"Model u prosjeku griješi oko {m['MAE']:.0f} $")

    pred = model.predict(X_test)
    table = pd.DataFrame({"stvarna": y_test.values, "predviđena": pred.round(0)})
    table["greška"] = (table["stvarna"] - table["predviđena"]).abs()
    print(table.head(10))
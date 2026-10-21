import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from src.data_cleaning import load_data, clean_data
from src.feature_engineering import add_features
from src.data_preprocessing import get_split
from src.model_training import build_pipeline
from src.model_evaluation import evaluate

MODELS = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=15, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

if __name__ == "__main__":
    df = add_features(clean_data(load_data()))
    X_train, X_test, y_train, y_test = get_split(df)

    results, fitted = [], {}
    for name, model in MODELS.items():
        print("Treniram:", name)
        pipe = build_pipeline(model).fit(X_train, y_train)
        fitted[name] = pipe
        results.append({"model": name, **evaluate(pipe, X_test, y_test)})

    res = pd.DataFrame(results).sort_values("MAE").round(3)
    print(res)
    res.to_csv("models/model_comparison.csv", index=False)

    best = res.iloc[0]["model"]
    joblib.dump(fitted[best], "models/car_price_model.joblib", compress=3)
    print("Finalni model:", best)
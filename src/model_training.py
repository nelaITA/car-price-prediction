import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from src.data_cleaning import load_data, clean_data
from src.feature_engineering import add_features
from src.data_preprocessing import get_preprocessor, get_split

def build_pipeline(model):
    return Pipeline([("preprocessor", get_preprocessor()), ("model", model)])

if __name__ == "__main__":
    df = add_features(clean_data(load_data()))
    X_train, X_test, y_train, y_test = get_split(df)
    pipe = build_pipeline(RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=42, n_jobs=-1))
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, "models/car_price_model.joblib", compress=3)
    print("Model sačuvan.")
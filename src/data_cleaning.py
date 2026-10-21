import pandas as pd

CAT_COLS = ["make", "model", "condition", "fuel_type", "color",
            "transmission", "drive_unit", "segment"]

def load_data(path="data/cars.csv"):
    return pd.read_csv(path)

def clean_data(df):
    df = df.copy()
    # 1. čisti nazivi kolona
    df = df.rename(columns={"mileage(kilometers)": "mileage_km",
                            "volume(cm3)": "volume_cm3"})
    # 2. duplikati
    df = df.drop_duplicates()
    # 3. standardizacija teksta (NaN ostaje NaN)
    for c in CAT_COLS:
        df[c] = df[c].str.strip().str.lower()
    # 4. nevalidne/ekstremne vrijednosti
    df = df[df["priceUSD"].between(300, 100_000)]
    df = df[df["year"].between(1970, 2019)]
    df = df[df["mileage_km"].between(0, 1_000_000)]
    df = df[df["volume_cm3"].isna() | df["volume_cm3"].between(600, 7000)]
    return df.reset_index(drop=True)

if __name__ == "__main__":
    raw = load_data()
    clean = clean_data(raw)
    print(raw.shape, "->", clean.shape)
CURRENT_YEAR = 2026

def add_features(df):
    df = df.copy()
    # starost automobila direktno utiče na cijenu
    df["car_age"] = CURRENT_YEAR - df["year"]
    # intenzitet korištenja: kilometri po godini starosti
    df["mileage_per_year"] = df["mileage_km"] / df["car_age"].clip(lower=1)
    # zapremina u litrima je čitljivija jedinica
    df["engine_volume_liters"] = df["volume_cm3"] / 1000
    # indikator veoma velike kilometraže
    df["is_high_mileage"] = (df["mileage_km"] > 300_000).astype(int)
    # year i volume_cm3 su sada redundantni
    return df.drop(columns=["year", "volume_cm3"])
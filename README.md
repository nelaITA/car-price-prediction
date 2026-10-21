# Car Price Prediction

Regresioni model koji predviđa cijenu polovnog automobila u dolarima (`priceUSD`) na osnovu njegovih karakteristika.

## Dataset
`data/cars.csv` sadrži oko 56.000 oglasa polovnih automobila sa kolonama: make, model, priceUSD, year, condition, mileage, fuel_type, volume, color, transmission, drive_unit, segment.

## Struktura projekta
```
car-price-prediction/
├── data/cars.csv
├── notebooks/01_eda.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── model_comparison.py
├── models/
│   ├── car_price_model.joblib
│   └── model_comparison.csv
├── README.md
└── requirements.txt
```

## Pokretanje
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m src.model_training
python -m src.model_evaluation
python -m src.model_comparison
```

## Čišćenje podataka
- preimenovane kolone `mileage(kilometers)` i `volume(cm3)`
- uklonjeni duplikati
- standardizovan tekst (mala slova, bez razmaka)
- uklonjene nevalidne vrijednosti: cijena van 300-100.000$, godina van 1970-2019, kilometraža preko 1.000.000 km, zapremina van 600-7000 cm3
- nedostajuće vrijednosti (drive_unit, segment, volume) popunjava imputer u pipeline-u

## Nove karakteristike
- `car_age`: starost automobila (starije vozilo je jeftinije)
- `mileage_per_year`: prosječna godišnja kilometraža (intenzitet korištenja)
- `engine_volume_liters`: zapremina motora u litrima
- `is_high_mileage`: indikator kilometraže preko 300.000 km

## Pretprocesiranje
- numeričke kolone: `SimpleImputer(median)` + `StandardScaler`
- kategorijske kolone: `SimpleImputer(most_frequent)` + `OneHotEncoder`
- podjela 80% trening / 20% test (`random_state=42`), isti skup za sve modele

## Rezultati

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest | 1090 | 2392 | 0.905 |
| Decision Tree | 1337 | 3058 | 0.845 |
| Gradient Boosting | 1463 | 2746 | 0.875 |
| Linear Regression | 2247 | 4256 | 0.700 |

## Finalni model
Izabran je **Random Forest Regressor** jer ima najmanju grešku (MAE) i najveći R² od svih testiranih modela.

Random Forest je podešen sa `min_samples_leaf=3` kako bi sačuvani model bio manji (ispod 100 MB, limit GitHuba).

**Tumačenje:** MAE od 1090$ znači da model u prosjeku griješi oko 1090 dolara. R² od 0.905 znači da model objašnjava oko 90% varijacije u cijenama. Linearna regresija je najslabija jer odnos između karakteristika i cijene nije linearan.

Primjeri predviđanja:

| Stvarna cijena | Predviđena cijena | Greška |
|---|---|---|
| 2200 | 2298 | 98 |
| 40000 | 41236 | 1236 |
| 3000 | 4137 | 1137 |
| 11600 | 11057 | 543 |
| 5000 | 4267 | 733 |
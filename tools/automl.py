import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def rf_and_gb(train, test, REF, restitution):
    # Charger les datasets
    train_df = pd.read_csv(train).fillna("")
    backtest_df = pd.read_csv(test).fillna("")


    # Séparer les features et labels
    X_train = train_df.drop(columns=REF)
    y_train = train_df[REF[-1]]
    X_backtest = backtest_df.drop(columns=REF)
    y_backtest = backtest_df[REF[-1]]
    ids = backtest_df[REF[0]]


    # 🔄 Encodage automatique des colonnes texte (get_dummies sur l'ensemble combiné)
    combined = pd.concat([X_train, X_backtest], axis=0)
    combined_encoded = pd.get_dummies(combined)

    # Re-séparation en X_train et X_backtest
    X_train_encoded = combined_encoded.iloc[:len(X_train), :]
    X_backtest_encoded = combined_encoded.iloc[len(X_train):, :]

    # Entraîner les modèles sans parallélisme
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=8)  # n_jobs by default = 1
    gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=3)

    print("⏳ Entraînement Random Forest...")
    rf_model.fit(X_train_encoded, y_train)

    print("⏳ Entraînement Gradient Boosting...")
    gb_model.fit(X_train_encoded, y_train)

    # Prédictions sur le backtest
    print("🔍 Prédictions...")
    rf_conf = rf_model.predict_proba(X_backtest_encoded)[:, 1]
    gb_conf = gb_model.predict_proba(X_backtest_encoded)[:, 1]

    # Résultats
    results = pd.DataFrame({
        REF[0]: ids,
        REF[-1]: y_backtest,
        "random_forest_confidence": rf_conf,
        "gradient_boosting_confidence": gb_conf
    })

    # Sauvegarder les résultats
    results.to_csv(restitution, index=False)
    print("✅ Résultats enregistrés dans '" + str(restitution) + "'")
import sys
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("> usage <train.csv : str> <test.csv : str> <REFS : str(s)-comma separated>")
        exit()
    rf_and_gb(sys.argv[1], sys.argv[2], sys.argv[3].split(","), "rf_gb_automl.csv")

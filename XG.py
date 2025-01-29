import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score
import phik

# ===================== 1. Caricare il dataset =====================
file_path = "data/NPHA-doctor-visits.csv"  # Modifica il percorso se necessario
df = pd.read_csv(file_path)

# Rinominare la colonna target per coerenza
df.rename(columns={'Number of Doctors Visited': 'Doctors_Visited'}, inplace=True)

# Separare feature e target
X = df.drop(columns=['Doctors_Visited'])
y = df['Doctors_Visited'] - 1  # 🔹 Ora le classi sono [0, 1, 2] invece di [1, 2, 3]

# ===================== 2. Analisi della Correlazione con Phi-K =====================
phik_corr = X.phik_matrix()
plt.figure(figsize=(12, 8))
sns.heatmap(phik_corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matrice di Correlazione Phi-K")
plt.show()

# ===================== 3. Stratified K-Fold Cross Validation =====================
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# ===================== 4. Definizione degli iperparametri per GridSearch =====================
param_grid_cat = {
    "iterations": [50, 100, 200],
    "learning_rate": [0.05, 0.1, 0.2],
    "depth": [4, 6, 8]
}

param_grid_rf = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10]
}

# ===================== 5. Dizionario dei modelli per GridSearch =====================
models = {
    "CatBoost": {
        "regressor": CatBoostRegressor(random_seed=42, verbose=0),
        "classifier": CatBoostClassifier(random_seed=42, verbose=0),
        "param_grid": param_grid_cat
    },
    "RandomForest": {
        "regressor": RandomForestRegressor(random_state=42),
        "classifier": RandomForestClassifier(random_state=42),
        "param_grid": param_grid_rf
    }
}

# ===================== 6. GridSearchCV per ottimizzazione =====================
best_params = {}
results = []

for model_name, model_dict in models.items():
    for task in ["regressor", "classifier"]:
        print(f"\n🔍 Ottimizzazione di {model_name} per {task}...")

        model = model_dict[task]
        param_grid = model_dict["param_grid"]

        grid_search = GridSearchCV(model, param_grid, cv=5,
                                   scoring="neg_mean_absolute_error" if task == "regressor" else "accuracy", n_jobs=-1,
                                   verbose=1)
        grid_search.fit(X, y)

        best_params[f"{model_name}_{task}"] = grid_search.best_params_
        best_model = grid_search.best_estimator_

        # Predizioni e metriche finali
        y_pred = best_model.predict(X)
        if task == "regressor":
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            results.append([model_name, "Regressione", mae, rmse, None, None])
        else:
            accuracy = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, average='weighted')
            results.append([model_name, "Classificazione", None, None, accuracy, f1])

# ===================== 7. Creazione DataFrame con i risultati =====================
df_results = pd.DataFrame(results, columns=["Modello", "Tipo", "MAE", "RMSE", "Accuracy", "F1-Score"])
print("\n===== RISULTATI OTTIMIZZATI =====")
print(df_results)

# ===================== 8. Plot dei risultati =====================
plt.figure(figsize=(12, 6))
sns.barplot(data=df_results, x="Modello", y="MAE", hue="Tipo")
plt.title("MAE per Modello")
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=df_results, x="Modello", y="RMSE", hue="Tipo")
plt.title("RMSE per Modello")
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=df_results, x="Modello", y="Accuracy", hue="Tipo")
plt.title("Accuracy per Modello")
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=df_results, x="Modello", y="F1-Score", hue="Tipo")
plt.title("F1-Score per Modello")
plt.show()

# ===================== 9. Importanza delle Feature =====================
final_models = {
    "CatBoost": CatBoostClassifier(**best_params["CatBoost_classifier"], random_seed=42, verbose=0),
    "RandomForest": RandomForestClassifier(**best_params["RandomForest_classifier"], random_state=42)
}

plt.figure(figsize=(12, 5))
for i, (model_name, final_model) in enumerate(final_models.items()):
    final_model.fit(X, y)

    plt.subplot(1, 2, i + 1)
    plt.title(f"Feature Importance - {model_name}")

    if model_name == "CatBoost":
        feature_importance = final_model.get_feature_importance()
        sns.barplot(x=feature_importance, y=X.columns)
    elif model_name == "RandomForest":
        feature_importance = final_model.feature_importances_
        sns.barplot(x=feature_importance, y=X.columns)

plt.tight_layout()
plt.show()

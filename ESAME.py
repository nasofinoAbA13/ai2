import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Caricamento dataset
df = pd.read_csv("data/NPHA-doctor-visits.csv")

#EXPLORATORY DATA ANALYSIS
print(df.head())
print(df.dtypes)
print(df.describe())


# Pulizia dei dati (-1 e -2 sono considerati valori mancanti)
df_cleaned = df.replace([-1, -2], np.nan).dropna()

#vediamo la distribuzione dei dati di ogni colonna (feature)
colonne = df_cleaned.columns.tolist()
for colonna in colonne:
    print(colonna)
    plt.figure(figsize = (12, 4))
    plt.subplot(1, 2, 1)
    df_cleaned[colonna].hist(grid=False)
    plt.xlabel(colonna)
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df_cleaned[colonna])
    plt.show()

##Trouble Sleeping è indicata come variabile binaria (0 e 1), ma nel dataset assume valori discreti tra 1 e 3,
#prima di eliminarla o trasformarla in binaria, analizzo la correlazione tra con il target:
# non c'è una differenza significativa nei valori di Trouble Sleeping tra chi ha 1, 2 o 3 visite dal medico
plt.figure(figsize=(6,4))
sns.boxplot(x=df_cleaned["Number of Doctors Visited"], y=df_cleaned["Trouble Sleeping"])
plt.title("Distribuzione di Trouble Sleeping rispetto al numero di visite mediche")
plt.xlabel("Numero di visite dal medico")
plt.ylabel("Trouble Sleeping")
plt.show()


# Calcolo della matrice di correlazione
correlation_matrix = df_cleaned.corr()

# Visualizzazione della matrice di correlazione
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
plt.title("Correlation Matrix")
plt.title("Correlation Matrix", fontsize=20)  # Aumenta la dimensione del titolo
plt.xticks(rotation=45, ha="right", fontsize=12)  # Ruota e ridimensiona le etichette
plt.yticks(fontsize=12)  # Dimensione delle etichette sull'asse Y
plt.tight_layout()  # Regola automaticamente gli spazi
plt.show()


# Combinazione di Physical Health e Mental Health
df_cleaned["Health Combined"] = df_cleaned["Phyiscal Health"] + df_cleaned["Mental Health"]

df_cleaned["Health Combined2"] = df_cleaned["Phyiscal Health"] + df_cleaned["Dental Health"]

"""
print("Correlazione di Health Combined:")
print(df_cleaned[["Health Combined", "Number of Doctors Visited"]].corr())

print("Correlazione di Health Combined2:")
print(df_cleaned[["Health Combined2", "Number of Doctors Visited"]].corr())
"""

# Variabili indipendenti e target HO TOLTO "AGE", HO TOLTO "TROUBLE SLEEPING"
target = "Number of Doctors Visited"
features = ['Phyiscal Health', 'Mental Health','Dental Health','Employment', 'Stress Keeps Patient from Sleeping',
       'Medication Keeps Patient from Sleeping',
       'Pain Keeps Patient from Sleeping',
       'Bathroom Needs Keeps Patient from Sleeping',
       'Uknown Keeps Patient from Sleeping', 'Prescription Sleep Medication', 'Race', 'Gender',
        'Health Combined','Health Combined2']

X = df_cleaned[features]
y = df_cleaned[target]

# Analisi della distribuzione delle classi target nel dataset originale
print("Distribuzione delle classi target nel dataset originale:")
print(y.value_counts())

# Grafico della distribuzione originale
plt.figure(figsize=(8, 6))
sns.countplot(x=y)
plt.title("Distribuzione delle classi (Originale)")
plt.xlabel("Numero di visite dal medico")
plt.ylabel("Conteggio")
plt.show()

#PREPARAZIONE DEI DATI:

# Preprocessing delle feature per renderle omogenee
ordinal_features = ["Phyiscal Health", "Mental Health", "Dental Health","Employment","Prescription Sleep Medication"]
binary_features = ["Stress Keeps Patient from Sleeping",
    "Medication Keeps Patient from Sleeping", 'Uknown Keeps Patient from Sleeping',
    "Pain Keeps Patient from Sleeping",  'Bathroom Needs Keeps Patient from Sleeping']

nominal_features = ["Race", "Gender"]

# Pipeline per la normalizzazione e l'encoding
preprocessor = ColumnTransformer(
    transformers=[
        ("ordinal", "passthrough", ordinal_features),  # Mantenere ordinali come numerici
        ("binary", "passthrough", binary_features),  # Mantenere binari come numerici
        ("nominal", OneHotEncoder(), nominal_features)  # Encoding per categorie nominali
    ]
)

# Standardizzo i valori
scaler = StandardScaler()
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("scaler", scaler)  # Applico lo scaling dopo l'elaborazione
])

# Applico il preprocessing
X_processed = pipeline.fit_transform(X)

# Split Train/Validation/Test
X_train, X_temp, y_train, y_temp = train_test_split(X_processed, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Identifico gli indici delle colonne categoriali
categorical_features = [features.index("Race"), features.index("Gender")]

# Bilanciamento con SMOTE-NC
smote_nc = SMOTENC(categorical_features=categorical_features, random_state=42)
X_train_balanced, y_train_balanced = smote_nc.fit_resample(X_train, y_train)

# Analisi della distribuzione delle classi dopo SMOTE-NC
print("Distribuzione delle classi sul training set dopo SMOTE-NC:")
print(pd.Series(y_train_balanced).value_counts())

# Grafico della distribuzione bilanciata
plt.figure(figsize=(8, 6))
sns.countplot(x=pd.Series(y_train_balanced))
plt.title("Distribuzione delle classi (Bilanciata con SMOTE-NC)")
plt.xlabel("Numero di visite dal medico")
plt.ylabel("Conteggio")
plt.show()


# tentativo con LOGISTIC REGRESSION:
# Ricerca iperparametrica
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'solver': ['lbfgs', 'newton-cg', 'saga'],
    'max_iter': [1000, 2000, 3000]
}
grid = GridSearchCV(
    LogisticRegression(class_weight="balanced", random_state=42),
    param_grid,
    cv=5,
    scoring="f1_macro"
)
grid.fit(X_train_balanced, y_train_balanced) #esecuzione del modello per ogni combinazione

# Migliori parametri trovati
print("Best Parameters:", grid.best_params_)
best_model = grid.best_estimator_

# Calcolo delle perdite finali
train_loss = log_loss(y_train_balanced, best_model.predict_proba(X_train_balanced))
val_loss = log_loss(y_val, best_model.predict_proba(X_val))
test_loss = log_loss(y_test, best_model.predict_proba(X_test))

# Stampa delle perdite finali
print(f"Train Loss: {train_loss:.4f}")
print(f"Validation Loss: {val_loss:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Classificazione e metriche
y_pred_test_best = best_model.predict(X_test)
print("Classification Report (Logistic Regression Ottimizzata):")
print(classification_report(y_test, y_pred_test_best))

# Matrice di confusione
conf_matrix = confusion_matrix(y_test, y_pred_test_best)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Test Set)")
plt.show()


"""""
#PROVO A IMPLEMENTARE ANCHE RANDOM FOREST, vedo se migliora qualcosa

# Faccio il tuning dei parametri, cioè scelgo quelli che funzionano meglio
param_grid = {
    "n_estimators": [50, 100],  # Ridotto per evitare troppi alberi
    "max_depth": [5, 10, 20],  # Ridotto per regolarizzare
    "min_samples_split": [5, 10],  # Maggiori valori per evitare overfitting
    "min_samples_leaf": [2, 4],  # Foglie con più campioni
    "class_weight": ["balanced", "balanced_subsample"]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="f1_macro", verbose=2, n_jobs=-1)
grid_search.fit(X_train_balanced, y_train_balanced)

best_rf = grid_search.best_estimator_

# Valuto con cross-validation più rigorosa
cv_scores = cross_val_score(best_rf, X_train_balanced, y_train_balanced, cv=10, scoring="f1_macro")  # 10-fold cross-validation
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean Cross-Validation Score: {np.mean(cv_scores):.4f}")

# Training e validazione
train_loss = log_loss(y_train_balanced, best_rf.predict_proba(X_train_balanced))
val_loss = log_loss(y_val, best_rf.predict_proba(X_val))
test_loss = log_loss(y_test, best_rf.predict_proba(X_test))

print(f"Migliori parametri: {grid_search.best_params_}")
print(f"Train Loss: {train_loss:.4f}")
print(f"Validation Loss: {val_loss:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Classificazione e metriche
y_pred_test = best_rf.predict(X_test)
print("Classification Report (Test Set):")
print(classification_report(y_test, y_pred_test))

# Matrice di confusione
conf_matrix = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Test Set)")
plt.show()


# Grafico delle perdite
estimators = [50, 100]
train_losses = []
val_losses = []

for n in estimators:
    rf_temp = RandomForestClassifier(n_estimators=n, random_state=42, class_weight="balanced")
    rf_temp.fit(X_train_balanced, y_train_balanced)
    train_losses.append(log_loss(y_train_balanced, rf_temp.predict_proba(X_train_balanced)))
    val_losses.append(log_loss(y_val, rf_temp.predict_proba(X_val)))

plt.figure(figsize=(8, 6))
plt.plot(estimators, train_losses, label="Training Loss", marker="o")
plt.plot(estimators, val_losses, label="Validation Loss", marker="o")
plt.xlabel("Number of Estimators")
plt.ylabel("Log Loss")
plt.title("Loss Curve for Random Forest")
plt.legend()
plt.show()

"""""
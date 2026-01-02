
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from catboost import Pool
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


file_path = "Dropout_Academic Success - Sheet1.csv"

df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "ankanhore545/dropout-or-academic-success",
  file_path,
)


X = df.drop(columns=['Target'])
y = df['Target']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Padronizando os recursos


model = CatBoostClassifier(
    loss_function="MultiClass",
    eval_metric="TotalF1",          # ou "MultiClass"
    random_seed=42,
    auto_class_weights="Balanced",  # balanceamento automático
    border_count=252,
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    early_stopping_rounds=50,
    verbose=100
)



le = LabelEncoder()
y_encoded = le.fit_transform(y)


X_train, X_val, y_train, y_val = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


train_pool = Pool(
    X_train,
    y_train,
    feature_names=list(X_train.columns)
)

val_pool = Pool(
    X_val,
    y_val,
    feature_names=list(X_val.columns)
)


model.fit(
    train_pool,
    eval_set=val_pool,
    use_best_model=True
)

y_pred = model.predict(X_val)


y_proba = model.predict_proba(X_val)

DROPOUT_CLASS = 0  # Dropout

dropout_probs = y_proba[:, DROPOUT_CLASS]

threshold = 0.30 #"Se a probabilidade de Dropout for maior que 30%, então classifique como Dropout, mesmo que outra classe tenha probabilidade maior."

y_pred_thresh = np.argmax(y_proba, axis=1)
y_pred_thresh[dropout_probs >= threshold] = DROPOUT_CLASS


cm = confusion_matrix(y_val, y_pred_thresh)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão – Threshold Ajustado")
plt.show()


feature_importance = model.get_feature_importance(
    type="PredictionValuesChange"
)

feat_imp = pd.Series(
    feature_importance,
    index=X_train.columns
).sort_values(ascending=False)

print("🔝 Importância das features (CatBoost):")
print(feat_imp.head(15))

plt.figure(figsize=(8, 6))
plt.barh(feat_imp.head(15).index[::-1], feat_imp.head(15).values[::-1])
plt.xlabel("Importância")
plt.ylabel("Feature")
plt.title("Importância das Features (CatBoost)")
plt.tight_layout()
plt.show()

precision, recall, f1, support = precision_recall_fscore_support(
    y_val,
    y_pred_thresh,
    labels=np.unique(y_val)
)

metrics_df = pd.DataFrame({
    "Classe": le.classes_,
    "Precisão": precision,
    "Recall": recall,
    "F1-score": f1,
    "Support": support
})

metrics_df

plt.figure(figsize=(8, 3))
plt.axis("off")

table = plt.table(
    cellText=np.round(metrics_df.iloc[:, 1:].values, 3),
    colLabels=metrics_df.columns[1:],
    rowLabels=metrics_df["Classe"],
    loc="center",
    cellLoc="center"
)

table.scale(1, 1.5)
table.auto_set_font_size(False)
table.set_fontsize(10)

plt.title("Métricas por Classe (Threshold Ajustado)", pad=20)
plt.show()

model.save_model("models/catboost_model.cbm")
X_val.to_csv("data/X_val.csv", index=False)
np.save("data/y_proba.npy", y_proba)

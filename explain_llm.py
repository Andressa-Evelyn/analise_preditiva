import pandas as pd
import numpy as np
import subprocess
from catboost import CatBoostClassifier, Pool


model = CatBoostClassifier()
model.load_model("models/catboost_model.cbm")

X_val = pd.read_csv("data/X_val.csv")
y_proba = np.load("data/y_proba.npy")

feature_names = list(X_val.columns)
DROPOUT_CLASS = 0


def build_prompt(df, proba):
    text = f"""
Você é um assistente educacional.

A probabilidade de evasão (Dropout) deste aluno é {proba:.1%}.

Principais fatores que influenciaram essa previsão:
"""
    for _, row in df.iterrows():
        direction = "aumenta" if row["impact"] > 0 else "reduz"
        text += f"- {row['feature']} = {row['value']} ({direction} o risco)\n"

    text += """
Gere uma explicação curta, clara e personalizada,
em linguagem simples e sem termos técnicos.
"""
    return text


i = 1  # índice do aluno

student = X_val.iloc[[i]]  # DataFrame com 1 linha
student_pool = Pool(student, feature_names=feature_names)


local_importance = model.get_feature_importance(
    data=student_pool,
    type="PredictionValuesChange"
)

explanation_df = pd.DataFrame({
    "feature": feature_names,
    "value": student.iloc[0].values,
    "impact": local_importance
}).sort_values(by="impact", key=abs, ascending=False).head(5)


prompt = build_prompt(
    explanation_df,
    y_proba[i, DROPOUT_CLASS]
)

result = subprocess.run(
    ["ollama", "run", "gemma:2b"],
    input=prompt,
    text=True,
    capture_output=True
)

print(result.stdout)

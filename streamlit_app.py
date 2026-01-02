import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import train_model


# visual theme
sns.set_theme(style="darkgrid")
sns.set_palette("deep")
FIGSIZE_DEFAULT = (5, 4)

SHOW_ONLY = [
    "confusion_matrix",
    "model_metrics",
    "prediction_probabilities",
    "feature_importance"
]

BASE = Path(__file__).parent
RESULTS_DIR = BASE / "notebooks" / "andressa" / "resultados"


st.set_page_config(layout="centered", page_title="Resultados - Andressa")
st.title("📁 Resultados — análise de evasão")

def should_show(path):
    name = path.name.lower()
    return any(key in name for key in SHOW_ONLY)


if not RESULTS_DIR.exists():
	st.error("Pasta de resultados não encontrada: %s" % RESULTS_DIR)
else:
	csv_files = sorted(RESULTS_DIR.glob("*.csv"))

	st.sidebar.header("Arquivos encontrados")
	file_names = [f.name for f in csv_files]
	file_choice = st.sidebar.selectbox("Escolha um CSV para visualizar individualmente", ["-- Selecionar --"] + file_names)
	

	def show_dataframe(df):
		st.dataframe(df)
		csv = df.to_csv(index=False).encode("utf-8")
		st.download_button("Baixar CSV", csv, file_name="export.csv", mime="text/csv")


	def _normalize_series(s):
		if s.sum() == 0:
			return s
		return s / s.sum()


	def aggregate_feature_importances(paths):
		cols = []
		frames = []
		for p in paths:
			name = p.name.lower()
			if "feature" in name or "importance" in name:
				try:
					df = pd.read_csv(p)
					lower = [c.lower() for c in df.columns]
					if "feature" in lower:
						fcol = df.columns[lower.index("feature")]
						imp_candidates = [c for c in df.columns if "importance" in c.lower() or "score" in c.lower() or "value" in c.lower()]
						if imp_candidates:
							icol = imp_candidates[0]
							ser = df.set_index(fcol)[icol].astype(float)
							ser = ser.groupby(ser.index).mean()
							ser = _normalize_series(ser)
							frames.append(ser.rename(p.name))
				except Exception:
					continue
		if not frames:
			return None
		allf = pd.concat(frames, axis=1).fillna(0)
		mean_imp = allf.mean(axis=1).sort_values(ascending=False)
		return mean_imp


	def try_plot_confusion_from_long(df_long):
		if set(["true_label", "predicted_label", "count"]).issubset(df_long.columns):
			pivot = df_long.pivot(index="true_label", columns="predicted_label", values="count").fillna(0).astype(int)
			fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
			sns.heatmap(pivot, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False, linewidths=0.5, annot_kws={"size": 8})
			ax.set_xlabel("Predito")
			ax.set_ylabel("Real")
			plt.tight_layout()
			st.pyplot(fig, use_container_width=False)
			return True
		return False
	
	def plot_confusion_matrix_from_square(df):
		try:
			mat = df.values
			if mat.ndim == 2 and mat.shape[0] == mat.shape[1]:
				fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
				sns.heatmap(mat.astype(int), annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False, linewidths=0.5, annot_kws={"size": 8})
				plt.tight_layout()
				st.pyplot(fig, width=False)
				return True
		except Exception:
			pass
		return False


	def plot_bar_series(series, title="", horizontal=False, top_n=30):
		if isinstance(series, pd.Series):
			ser = series.copy()
		else:
			ser = pd.Series(series)
		ser = ser.dropna()
		ser = ser.sort_values(ascending=False).head(top_n)
		if horizontal:
			fig, ax = plt.subplots(figsize=(6, max(2, len(ser) * 0.25)))
			ser.plot(kind="barh", ax=ax, color=sns.color_palette("deep", n_colors=len(ser)))
			ax.invert_yaxis()
		else:
			fig, ax = plt.subplots(figsize=(6, 3))
			ser.plot(kind="bar", ax=ax, color=sns.color_palette("deep", n_colors=len(ser)))
		ax.set_title(title)
		plt.tight_layout()
		st.pyplot(fig, use_container_width=False)


	if file_choice and file_choice != "-- Selecionar --":
		path = RESULTS_DIR / file_choice
		try:
			df = pd.read_csv(path)
		except Exception as e:
			st.error(f"Erro ao ler o arquivo: {e}")
		else:
			st.subheader(file_choice)

			name_lower = file_choice.lower()
			
			if "confusion_matrix" not in name_lower:
				show_dataframe(df)

			if "confusion_matrix" in name_lower:
				plotted = try_plot_confusion_from_long(df)
				if not plotted:
					plotted = plot_confusion_matrix_from_square(df)
				if not plotted:
					st.info("Formato de matriz de confusão não reconhecido.")
				st.stop()

			if "prediction_probabilities_summary" in name_lower or "probabilities" in name_lower:
				if "mean_probability" in df.columns and "class" in df.columns:
					df2 = df.set_index("class")["mean_probability"]
					plot_bar_series(df2, "Probabilidade média por classe")

			if "model_metrics" in name_lower or "metrics" in name_lower:
				metric_cols = [c for c in df.columns if c.lower() not in ("model", "index")]
				if "Model" in df.columns or "model" in df.columns:
					idx = "Model" if "Model" in df.columns else "model"
					df_plot = df.set_index(idx)[metric_cols]
					st.subheader("Métricas por modelo")
					for c in df_plot.columns:
						plot_bar_series(df_plot[c], f"{c}")

			if "feature_importance" in name_lower or "feature" in name_lower:
				possible_cols = [c.lower() for c in df.columns]
				if "feature" in possible_cols and ("importance" in possible_cols or "score" in possible_cols):
					feat_col = df.columns[possible_cols.index("feature")]
					imp_col_candidates = [c for c in df.columns if "importance" in c.lower() or "score" in c.lower()]
					if imp_col_candidates:
						imp_col = imp_col_candidates[0]
						df_sorted = df.sort_values(imp_col, ascending=False).head(30).set_index(feat_col)[imp_col]
						plot_bar_series(df_sorted, "Top features por importância")


	
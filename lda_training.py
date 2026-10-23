from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import pandas as pd
from joblib import Memory
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from tsfresh_feature_selector import TSFreshFeatureSelector


def main():
	base_dir = Path(__file__).resolve().parent
	features_path = base_dir / "training_dataset/X_features_tsfresh.parquet"
	labels_path = base_dir / "training_dataset/y_labels.csv"

	X = pd.read_parquet(features_path)
	labels = pd.read_csv(labels_path).set_index("id")

	if "classe" not in labels.columns:
		raise ValueError("O arquivo y_labels.csv precisa conter a coluna 'classe'.")
	if not X.index.is_unique or not labels.index.is_unique:
		raise ValueError("Os identificadores das amostras precisam ser únicos.")
	if set(X.index) != set(labels.index):
		raise ValueError("Os IDs do parquet e do CSV de rótulos não correspondem.")

	y = labels.loc[X.index, "classe"]
	if y.isna().any():
		raise ValueError("Há rótulos ausentes em y_labels.csv.")
	if X.isna().any().any():
		raise ValueError(
			"A matriz de features contém valores ausentes; aplique imputação antes do treino."
		)

	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	pipeline = Pipeline(
		steps=[
			("feature_selection", TSFreshFeatureSelector(fdr_level=0.05, n_jobs=1)),
			("classifier", LinearDiscriminantAnalysis()),
		]
	)
	shrinkage_values = [value / 10 for value in range(11)] + ["auto"]
	parameter_grid = [
		{"classifier__solver": ["svd"]},
		{
			"classifier__solver": ["lsqr", "eigen"],
			"classifier__shrinkage": shrinkage_values,
		},
	]

	with TemporaryDirectory(prefix="tsfresh_lda_cache_") as cache_dir:
		pipeline.memory = Memory(location=cache_dir, verbose=0)
		search = GridSearchCV(
			estimator=pipeline,
			param_grid=parameter_grid,
			scoring="accuracy",
			cv=cv,
			refit=True,
			n_jobs=-1,
			return_train_score=False,
			error_score=float("nan"),
		)
		search.fit(X, y)

	best_pipeline = search.best_estimator_
	selected_columns = best_pipeline.named_steps["feature_selection"].selected_columns_

	model_path = base_dir / "LDA/best_model.joblib"
	selected_path = base_dir / "LDA/tsfresh_selected_features.csv"
	results_path = base_dir / "LDA/grid_search_results.csv"

	joblib.dump(best_pipeline, model_path)
	pd.Series(selected_columns, name="feature").to_csv(selected_path, index=False)
	pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(
		results_path, index=False
	)

	print(f"Melhor acurácia média (CV 5 dobras): {search.best_score_:.4f}")
	print(f"Melhores hiperparâmetros: {search.best_params_}")
	print(f"Features selecionadas no modelo final: {len(selected_columns)}")
	print(f"Modelo salvo em: {model_path}")
	print(f"Features selecionadas salvas em: {selected_path}")
	print(f"Resultados da busca salvos em: {results_path}")


if __name__ == "__main__":
	main()
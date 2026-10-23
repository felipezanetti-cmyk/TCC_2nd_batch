from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


def main():
	base_dir = Path(__file__).resolve().parent
	features_path = base_dir / "test_dataset/X_features_tsfresh.parquet"
	labels_path = base_dir / "test_dataset/y_labels.csv"
	model_paths = {
		"Random Forest": base_dir / "random_forest/best_model.joblib",
		"LDA": base_dir / "LDA/best_model.joblib",
	}

	X_test = pd.read_parquet(features_path)
	labels = pd.read_csv(labels_path).set_index("id")

	if "classe" not in labels.columns:
		raise ValueError("O arquivo y_labels.csv precisa conter a coluna 'classe'.")
	if not X_test.index.is_unique or not labels.index.is_unique:
		raise ValueError("Os identificadores das amostras de teste precisam ser únicos.")
	if set(X_test.index) != set(labels.index):
		raise ValueError("Os IDs do parquet e do CSV de rótulos não correspondem.")
	if not np.isfinite(X_test.to_numpy()).all():
		raise ValueError("A matriz de teste contém valores ausentes ou infinitos.")

	y_test = labels.loc[X_test.index, "classe"]
	if set(y_test.unique()) != {0, 1}:
		raise ValueError("Os rótulos de teste devem conter as classes 0 (nominal) e 1 (falha).")

	fault_ids = y_test.index[y_test == 1].sort_values().tolist()
	fault_case_sizes = {"Sutil": 40, "Media": 40, "Severa": 40}
	if sum(fault_case_sizes.values()) != len(fault_ids):
		raise ValueError(
			"A quantidade de falhas não corresponde aos grupos definidos "
			"(40 sutis, 40 médias e 40 severas). Atualize fault_case_sizes."
		)

	fault_cases = pd.Series("Nominal", index=y_test.index, dtype="object")
	start = 0
	for case_name, case_size in fault_case_sizes.items():
		case_ids = fault_ids[start : start + case_size]
		fault_cases.loc[case_ids] = case_name
		start += case_size

	metric_rows = []
	confusion_rows = []
	fault_case_metric_rows = []
	fault_case_confusion_rows = []
	class_labels = [0, 1]
	class_names = {0: "Nominal", 1: "Falha"}

	for model_name, model_path in model_paths.items():
		model = joblib.load(model_path)
		predictions = model.predict(X_test)
		matrix = confusion_matrix(y_test, predictions, labels=class_labels)
		tn, fp, fn, tp = matrix.ravel()

		metric_rows.append(
			{
				"modelo": model_name,
				"FDR": tp / (tp + fn) if tp + fn else 0.0,
				"FAR": fp / (fp + tn) if fp + tn else 0.0,
				"F1_falha": f1_score(
					y_test, predictions, pos_label=1, zero_division=0
				),
			}
		)

		for row_index, actual_class in enumerate(class_labels):
			confusion_rows.append(
				{
					"modelo": model_name,
					"classe_real": class_names[actual_class],
					"classe_real_codigo": actual_class,
					"previsto_nominal_0": int(matrix[row_index, 0]),
					"previsto_falha_1": int(matrix[row_index, 1]),
				}
			)

		predictions_by_id = pd.Series(predictions, index=X_test.index)
		for case_name, case_size in fault_case_sizes.items():
			case_mask = (y_test == 0) | (fault_cases == case_name)
			case_y = y_test.loc[case_mask]
			case_predictions = predictions_by_id.loc[case_mask]
			case_matrix = confusion_matrix(case_y, case_predictions, labels=class_labels)
			case_tn, case_fp, case_fn, case_tp = case_matrix.ravel()

			fault_case_metric_rows.append(
				{
					"modelo": model_name,
					"caso_falha": case_name,
					"n_amostras_falha": case_size,
					"falhas_detectadas": int(case_tp),
					"falhas_nao_detectadas": int(case_fn),
					"FDR": case_tp / (case_tp + case_fn) if case_tp + case_fn else 0.0,
					"FAR": case_fp / (case_fp + case_tn) if case_fp + case_tn else 0.0,
					"F1": f1_score(case_y, case_predictions, pos_label=1, zero_division=0),
				}
			)

			for row_index, actual_group in enumerate(["Nominal", case_name]):
				fault_case_confusion_rows.append(
					{
						"modelo": model_name,
						"caso_falha": case_name,
						"grupo_real": actual_group,
						"previsto_nominal_0": int(case_matrix[row_index, 0]),
						"previsto_falha_1": int(case_matrix[row_index, 1]),
					}
				)

		classifier = model.named_steps["classifier"]
		params = classifier.get_params()
		if model_name == "Random Forest":
			config = {
				"n_estimators": params["n_estimators"],
				"max_depth": params["max_depth"],
				"min_samples_split": params["min_samples_split"],
				"min_samples_leaf": params["min_samples_leaf"],
			}
		else:
			config = {"solver": params["solver"], "shrinkage": params["shrinkage"]}
		print(f"Configuração carregada ({model_name}): {config}")

	evaluation_dir = base_dir / "evaluation_results"

	metrics_path = evaluation_dir / "test_model_metrics.csv"
	confusion_path = evaluation_dir / "test_confusion_matrices.csv"
	fault_case_metrics_path = evaluation_dir / "test_fault_intensity_metrics.csv"
	fault_case_confusion_path = evaluation_dir / "test_fault_intensity_confusion_matrices.csv"
	pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
	pd.DataFrame(confusion_rows).to_csv(confusion_path, index=False)
	pd.DataFrame(fault_case_metric_rows).to_csv(fault_case_metrics_path, index=False)
	pd.DataFrame(fault_case_confusion_rows).to_csv(fault_case_confusion_path, index=False)

	print("\nMétricas no conjunto de teste:")
	print(pd.DataFrame(metric_rows).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
	print("\nMétricas por intensidade de falha (comparação com amostras nominais):")
	print(
		pd.DataFrame(fault_case_metric_rows).to_string(
			index=False, float_format=lambda value: f"{value:.4f}"
		)
	)
	print(f"\nMétricas salvas em: {metrics_path}")
	print(f"Matrizes de confusão salvas em: {confusion_path}")
	print(f"Métricas por intensidade salvas em: {fault_case_metrics_path}")
	print(f"Matrizes por intensidade salvas em: {fault_case_confusion_path}")


if __name__ == "__main__":
	main()
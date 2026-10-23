from sklearn.base import BaseEstimator, TransformerMixin
from tsfresh import select_features


class TSFreshFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, fdr_level=0.05, n_jobs=1):
        self.fdr_level = fdr_level
        self.n_jobs = n_jobs

    def fit(self, X, y):
        selected = select_features(
            X,
            y,
            fdr_level=self.fdr_level,
            n_jobs=self.n_jobs,
            show_warnings=False,
        )
        self.selected_columns_ = selected.columns.tolist()
        if not self.selected_columns_:
            raise ValueError(
                "O TSFresh não selecionou nenhuma feature com o nível de FDR definido."
            )
        return self

    def transform(self, X):
        missing_columns = set(self.selected_columns_) - set(X.columns)
        if missing_columns:
            raise ValueError("A matriz recebida não contém todas as features selecionadas.")
        return X.loc[:, self.selected_columns_]
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC


def build_model() -> Pipeline:
    return Pipeline([
        ("prep", ColumnTransformer([("num", StandardScaler(), NUMERIC),
                                    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])),
        ("clf", LogisticRegression(max_iter=2000, C=0.5)),
    ])

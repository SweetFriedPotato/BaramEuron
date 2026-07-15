import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self, scale=False, strategy="median"):
        self.imputer=SimpleImputer(strategy=strategy)
        self.scaler=StandardScaler() if scale else None
        self.feature_names_=None
    def fit(self,x):
        self.feature_names_=list(getattr(x, "columns", [f"x{i}" for i in range(np.asarray(x).shape[1])]))
        z=self.imputer.fit_transform(x)
        if self.scaler is not None: self.scaler.fit(z)
        return self
    def transform(self,x):
        z=self.imputer.transform(x); return self.scaler.transform(z) if self.scaler is not None else z
    def fit_transform(self,x): return self.fit(x).transform(x)
    def save(self,path): joblib.dump(self,path)
    @classmethod
    def load(cls,path): return joblib.load(path)

def _as_frame(x):
    if isinstance(x, pd.DataFrame):
        return x
    return pd.DataFrame(x)

def fit_model_preprocessor(train_x, *others, mode="tree", config=None):
    cfg = (config or {}).get("preprocessing", {}).get(mode, {})
    if mode not in {"tree", "neural"}:
        raise ValueError("mode must be 'tree' or 'neural'")
    scale = bool(cfg.get("scale", mode == "neural"))
    strategy = cfg.get("imputer_strategy", "median")
    preprocessor = Preprocessor(scale=scale, strategy=strategy).fit(_as_frame(train_x))
    transformed_train = preprocessor.transform(_as_frame(train_x))
    transformed_others = tuple(preprocessor.transform(_as_frame(x)) for x in others)
    return preprocessor, transformed_train, *transformed_others, preprocessor.feature_names_

def fit_tree_preprocessor(train_x, *others, config=None):
    return fit_model_preprocessor(train_x, *others, mode="tree", config=config)

def fit_neural_preprocessor(train_x, *others, config=None):
    return fit_model_preprocessor(train_x, *others, mode="neural", config=config)

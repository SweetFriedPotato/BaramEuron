import joblib
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from .base import BaseModel

class SklearnModel(BaseModel):
    def __init__(self,name="random_forest",**params):
        classes={"random_forest":RandomForestRegressor,"extra_trees":ExtraTreesRegressor}
        if name not in classes: raise ValueError(f"Unknown sklearn model: {name}")
        self.name=name; self.model=classes[name](**params)
    def fit(self,train_data,train_target,valid_data=None,valid_target=None): self.model.fit(train_data,train_target); return self
    def predict(self,data): return self.model.predict(data)
    def save(self,path): joblib.dump(self,path)
    @classmethod
    def load(cls,path): return joblib.load(path)


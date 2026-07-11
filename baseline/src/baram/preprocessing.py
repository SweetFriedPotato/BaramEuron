import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self, scale=False): self.imputer=SimpleImputer(strategy="median"); self.scaler=StandardScaler() if scale else None
    def fit(self,x):
        z=self.imputer.fit_transform(x)
        if self.scaler is not None: self.scaler.fit(z)
        return self
    def transform(self,x):
        z=self.imputer.transform(x); return self.scaler.transform(z) if self.scaler is not None else z
    def fit_transform(self,x): return self.fit(x).transform(x)
    def save(self,path): joblib.dump(self,path)
    @classmethod
    def load(cls,path): return joblib.load(path)


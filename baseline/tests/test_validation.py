import pandas as pd
from baram.validation import time_split
from baram.preprocessing import Preprocessor, fit_neural_preprocessor, fit_tree_preprocessor
def test_validation_is_future():
    t=pd.date_range("2022-01-01",periods=3*365*24,freq="h"); tr,va=time_split(t,"kpx_group_1"); assert t[tr].max()<t[va].min()
def test_preprocessor_train_only():
    tr=pd.DataFrame({"x":[1.,None,3.]}); te=pd.DataFrame({"x":[1000.,None]}); p=Preprocessor(True).fit(tr); assert p.imputer.statistics_[0]==2; p.transform(te); assert p.imputer.statistics_[0]==2

def test_tree_and_neural_preprocessor_contracts():
    train=pd.DataFrame({"x":[1.,None,3.]})
    valid=pd.DataFrame({"x":[1000.,None]})
    tree, tree_train, tree_valid, tree_names = fit_tree_preprocessor(train, valid)
    neural, neural_train, neural_valid, neural_names = fit_neural_preprocessor(train, valid)
    assert tree.scaler is None
    assert neural.scaler is not None
    assert tree.imputer.statistics_[0] == 2
    assert neural.imputer.statistics_[0] == 2
    assert tree_names == ["x"]
    assert neural_names == ["x"]
    assert tree_train.shape == neural_train.shape == (3, 1)
    assert tree_valid.shape == neural_valid.shape == (2, 1)

import pandas as pd
from baram.validation import time_split
from baram.preprocessing import Preprocessor
def test_validation_is_future():
    t=pd.date_range("2022-01-01",periods=3*365*24,freq="h"); tr,va=time_split(t,"kpx_group_1"); assert t[tr].max()<t[va].min()
def test_preprocessor_train_only():
    tr=pd.DataFrame({"x":[1.,None,3.]}); te=pd.DataFrame({"x":[1000.,None]}); p=Preprocessor(True).fit(tr); assert p.imputer.statistics_[0]==2; p.transform(te); assert p.imputer.statistics_[0]==2


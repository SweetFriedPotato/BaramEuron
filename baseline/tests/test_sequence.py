import numpy as np,pandas as pd
from baram.features.sequence import make_sequences
def test_sequence_shape_and_no_future():
    t=pd.date_range("2024-01-01",periods=30,freq="h"); d=pd.DataFrame({"forecast_kst_dtm":t,"value":np.arange(30)})
    x,kept=make_sequences(d,t,24); assert x.shape==(7,24,1); assert np.all(x[:,-1,0]==np.arange(23,30)); assert kept[0]==t[23]


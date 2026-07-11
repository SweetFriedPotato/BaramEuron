import numpy as np
from baram.data import load_sample_submission
from baram.constants import TARGETS
from baram.submission import create_submission
def test_submission_contract(cfg):
    sample=load_sample_submission(cfg); out=create_submission(sample,{t:np.zeros(len(sample)) for t in TARGETS}); assert len(out)==8760; assert list(out)==list(sample)


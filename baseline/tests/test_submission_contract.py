import numpy as np

from baram.constants import TARGETS
from baram.data import load_sample_submission
from baram.submission import create_submission, validate_submission_contract


def test_submission_schema_matches_sample(cfg):
    sample = load_sample_submission(cfg)
    predictions = {target: np.zeros(len(sample), dtype=float) for target in TARGETS}
    submission = create_submission(sample, predictions)
    assert list(submission.columns) == list(sample.columns)
    assert len(submission) == 8760
    assert validate_submission_contract(submission, sample)

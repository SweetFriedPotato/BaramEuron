import numpy as np

from baram.constants import TARGETS
from baram.data import load_sample_submission
from baram.submission import create_submission, validate_submission_contract
from experiments.exp02_daily_tcn_scada_aux.src.data_contract import baseline_config
from experiments.exp03_official_score_calibration.src.train_variants import official_validation_score


def test_exp04_reuses_official_scorer_behavior():
    target = np.full((2, 24, 3), 0.5, dtype=np.float32)
    mask = np.ones_like(target, dtype=bool)
    score, one_minus_nmae, ficr, groups = official_validation_score(target, target, mask)
    assert score == 1.0
    assert one_minus_nmae == 1.0
    assert ficr == 1.0
    assert len(groups) == 3


def test_submission_contract(tmp_path):
    sample = load_sample_submission(baseline_config())
    values = {target: np.linspace(0.0, 1.0, len(sample)) for target in TARGETS}
    submission = create_submission(sample, values, tmp_path / "submission.csv")
    validate_submission_contract(submission, sample)
    assert len(submission) == 8760
    assert not submission["forecast_kst_dtm"].duplicated().any()
    assert np.isfinite(submission[TARGETS].to_numpy()).all()

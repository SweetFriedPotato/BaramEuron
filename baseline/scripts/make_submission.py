#!/usr/bin/env python3
# Full training and prediction are intentionally atomic so preprocessing and model artifacts cannot drift.
exec(open(__file__.replace("make_submission.py","train_full.py"),encoding="utf-8").read())


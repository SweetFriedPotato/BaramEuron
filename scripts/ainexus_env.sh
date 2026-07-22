#!/usr/bin/env bash

# Shared runtime environment for the EURAM AI NEXUS session.
BARAM_ROOT="${BARAM_ROOT:-/home/work/baram/Baram}"
BARAM_PYTHON_PACKAGES="${BARAM_PYTHON_PACKAGES:-/home/work/baram/python_packages}"

export BARAM_ROOT
export BARAM_PYTHON_PACKAGES
export PYTHONPATH="${BARAM_PYTHON_PACKAGES}:${BARAM_ROOT}/baseline/src:${BARAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONUNBUFFERED=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

cd "${BARAM_ROOT}"

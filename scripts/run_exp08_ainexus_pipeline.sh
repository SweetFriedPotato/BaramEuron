#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ainexus_env.sh"

OUTPUT_ROOT="${BARAM_ROOT}/experiments/exp08_scada_hubwind_pretraining/outputs"
LOG_ROOT="${OUTPUT_ROOT}/logs"
MODULE="experiments.exp08_scada_hubwind_pretraining.src.run_experiment"
mkdir -p "${LOG_ROOT}"

run_phase() {
    local name="$1"
    shift
    local log_path="${LOG_ROOT}/${name}.log"
    local exit_path="${LOG_ROOT}/${name}.exit"
    echo "[$(date --iso-8601=seconds)] starting ${name}" | tee -a "${log_path}"
    set +e
    python3 -u -m "${MODULE}" "$@" 2>&1 | tee -a "${log_path}"
    local code=${PIPESTATUS[0]}
    set -e
    printf '%s\n' "${code}" > "${exit_path}"
    echo "[$(date --iso-8601=seconds)] finished ${name} exit=${code}" | tee -a "${log_path}"
    if (( code != 0 )); then
        return "${code}"
    fi
}

if [[ -n "${WAIT_FOR_STAGE1_PID:-}" ]]; then
    echo "Waiting for existing Stage-1 seed-42 PID ${WAIT_FOR_STAGE1_PID}."
    while kill -0 "${WAIT_FOR_STAGE1_PID}" 2>/dev/null; do
        sleep 30
    done
    checkpoint_count=$(find "${OUTPUT_ROOT}/checkpoints/stage1" -name '*.pt' 2>/dev/null | wc -l)
    if [[ ! -f "${OUTPUT_ROOT}/stage1_selection.json" || "${checkpoint_count}" -lt 32 ]]; then
        echo "Existing Stage-1 seed-42 run did not complete (${checkpoint_count}/32 checkpoints)." >&2
        exit 11
    fi
else
    run_phase stage1_seed42 --phase stage1 --seed 42
fi

run_phase stage1_seed52 --phase stage1 --seed 52
run_phase stage1_seed62 --phase stage1 --seed 62
run_phase stage2_seed42 --phase stage2 --seed 42
run_phase stage2_seed52 --phase stage2 --seed 52
run_phase stage2_seed62 --phase stage2 --seed 62
run_phase finalize --phase finalize
run_phase report --phase report --run-id "ainexus_$(date +%Y%m%d_%H%M%S)"

echo "[$(date --iso-8601=seconds)] Exp08 AI NEXUS pipeline complete." | tee "${LOG_ROOT}/pipeline.complete"

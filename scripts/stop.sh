#!/usr/bin/env bash
# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Stop any in-flight orchestrator run. Kept in its own file on purpose: if the
# pkill patterns appeared on the invoking shell's command line, pkill -f would
# match that shell and kill its own parent.
# Every orchestrator script, not just the ones that existed when this was
# written: two of these write the same results file, so a stale run left alive
# alongside a new one can silently overwrite its output.
for s in validate_pipeline sweep_coupling optimize.py compare_to_reference \
         compare_thermal_backends predict_error gradient_map_sweep \
         probe_startpoint; do
    pkill -9 -f "$s" 2>/dev/null || true
done
sleep 1

# SIGKILL skips ColdPlate.__exit__, so the served Tesseract containers survive
# their driver and pile up -- 15 of them at one point, all competing for CPU
# and making every subsequent run look mysteriously slow. Reap them here.
ids=$(docker ps -q --filter ancestor=stokes_brinkman \
                   --filter ancestor=thermal_advdiff \
                   --filter ancestor=material_map 2>/dev/null)
all=$(docker ps -q 2>/dev/null)
if [ -n "$all" ]; then
    docker rm -f $all >/dev/null 2>&1 || true
fi
echo "stopped; containers remaining: $(docker ps -q 2>/dev/null | wc -l)"

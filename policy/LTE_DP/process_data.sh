#!/usr/bin/env bash
set -Eeuo pipefail
exec python -m XPolicyLab.policy.LTE_DP.data "$@"

#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper to keep backwards compatibility with the original interface.
#
# The old evaluate.bash contained a lot of branching and duplicated CLI building.
# We now delegate to a python implementation that also supports parallelism.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

exec python "$SCRIPT_DIR/run_pipeline.py" "$@"

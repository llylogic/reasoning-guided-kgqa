#!/usr/bin/env bash
set -eu
exec python "$(dirname "$0")/launch_stage.py" 2 "$@"

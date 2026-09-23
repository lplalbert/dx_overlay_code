#!/usr/bin/env bash
set -euo pipefail

echo "[DEPRECATED] build_gcc.sh now delegates to the canonical Windows release build."
echo "             Use build_release.bat directly for all supported builds."

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
batch_path="$(cygpath -w "$script_dir/build_release.bat")"
cmd.exe /d /s /c "\"$batch_path\" --no-pause"

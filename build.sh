#!/usr/bin/env bash
# Windows x64 build entry point. Run with Git Bash.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"
install_dependencies=0
package_args=()
while (($#)); do
    case "$1" in
        --install-deps) install_dependencies=1; package_args+=(--install-deps) ;;
        --skip-build) package_args+=(--skip-build) ;;
        -h|--help)
            echo 'Usage: bash build.sh [--install-deps] [--skip-build]'
            echo 'Uses .venv/Scripts/python.exe; outputs cpp/build, model/build, build/windows.'
            echo '--install-deps: create .venv with uv if missing, then install runtime/build dependencies.'
            echo '--skip-build: reuse compiled outputs; still run package validation and parity tests.'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done
case "$(uname -s)" in
    MINGW*|MSYS*) ;;
    *) echo 'Use Git Bash on Windows to build this Windows x64 package.' >&2; exit 1 ;;
esac
if [[ ! -f .venv/Scripts/python.exe ]]; then
    if ((install_dependencies)); then
        uv venv --python 3.11 --seed .venv
    else
        echo 'Missing .venv. Run: bash build.sh --install-deps' >&2
        exit 1
    fi
fi
python="$repo_root/.venv/Scripts/python.exe"
exec "$python" "$repo_root/cpp/tools/build_windows_package.py" "${package_args[@]}"

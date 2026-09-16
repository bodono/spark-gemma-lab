#!/usr/bin/env bash
# Extract a verified Ubuntu development package without root or system changes.
set -euo pipefail
[[ $(uname -m) == aarch64 ]] || { echo 'Expected ARM64' >&2; exit 2; }
devroot="$HOME/.local/share/spark-gemma-lab/python-dev"
package=libpython3.12-dev_3.12.3-1ubuntu0.17_arm64.deb
expected=945ad3f651f683c11e72d19d804b1cff097308dcffa01057635cba038f875164
[[ $(python3 -c 'import platform; print(platform.python_version())') == 3.12.3 ]] || { echo 'These headers target Python 3.12.3' >&2; exit 2; }
mkdir -p "$devroot"
if [[ ! -f "$devroot/$package" ]]; then
 curl --fail --location --retry 3 --connect-timeout 20 --max-time 180 \
  "https://ports.ubuntu.com/ubuntu-ports/pool/main/p/python3.12/$package" \
  --output "$devroot/$package.part"
 mv "$devroot/$package.part" "$devroot/$package"
fi
printf '%s  %s\n' "$expected" "$devroot/$package" | sha256sum --check -
# Only include headers are extracted; no library or system package is installed.
dpkg-deb --fsys-tarfile "$devroot/$package" | tar -x -C "$devroot" ./usr/include
export CPATH="$devroot/usr/include/python3.12:$devroot/usr/include${CPATH:+:$CPATH}"
printf '#include <Python.h>\nint main(void) { return PY_MAJOR_VERSION != 3; }\n' | cc -x c -fsyntax-only -
echo 'Python 3.12.3 headers verified and C compilation check passed.'

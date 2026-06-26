#!/usr/bin/env bash
# Reproducible build of scikit-survival when no binary wheel is available for the
# platform (e.g. aarch64 / specific numpy ABI). Standard installation is simply:
#
#     pip install -r requirements.txt
#
# If that fails to find a scikit-survival wheel, build from source as follows.
set -euo pipefail

VER=0.25.0
pip install --break-system-packages -q Cython setuptools-scm "numpy>=2" "scikit-learn~=1.7.0" ecos osqp joblib

WORK=$(mktemp -d)
curl -sL "https://files.pythonhosted.org/packages/source/s/scikit-survival/scikit_survival-${VER}.tar.gz" \
  -o "$WORK/sksurv.tar.gz" || \
  pip download --no-binary :all: --no-deps "scikit-survival==${VER}" -d "$WORK"
mkdir -p "$WORK/src"
tar xzf "$WORK"/scikit_survival-*.tar.gz -C "$WORK/src" --strip-components=1

cd "$WORK/src"
export SETUPTOOLS_SCM_PRETEND_VERSION=$VER
# In-place compile of the Cython/C++ extensions (resumable across invocations).
python3 setup.py build_ext --inplace -j "$(nproc)"

# Copy the built package into the user site-packages.
SP=$(python3 -c "import site; print(site.getusersitepackages())")
mkdir -p "$SP"
rm -rf "$SP/sksurv"
cp -r sksurv "$SP/sksurv"

python3 -c "from sksurv.ensemble import RandomSurvivalForest; from sksurv.metrics import concordance_index_ipcw; print('scikit-survival ready')"

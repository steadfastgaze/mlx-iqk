#!/bin/zsh
# Build the vendored codecs into the shared libraries the Python encode shim
# loads with ctypes: the IQ2_K / IQ2_KS pair and the dense-member set
# (IQ4_KS, IQ4_K, IQ5_K, IQ6_K). Nothing outside this directory is written.
#
# Optimisation level matters for build time only. The codecs are integer and
# float32 scalar code with no reassociation-sensitive reductions guarded by
# fast-math, so -O3 and -O0 produce the same bytes; the tests assert that
# against ik_llama's own library when a checkout is present.
set -e
HERE=${0:a:h}
OUT=${HERE}/libiqk_iq2.dylib
c++ -O3 -std=c++17 -fPIC -shared -o "${OUT}" "${HERE}/iqk_iq2.cpp"
echo "built ${OUT}"
OUT=${HERE}/libiqk_dense.dylib
c++ -O3 -std=c++17 -fPIC -shared -o "${OUT}" "${HERE}/iqk_dense.cpp"
echo "built ${OUT}"

#!/bin/zsh
# Build the vendored IQ1_S_R4 codec into a shared library the Python encode
# shim loads with ctypes. Nothing outside this directory is written.
#
# Optimisation level matters for build time only. The codec is integer and
# float32 scalar code with no reassociation-sensitive reductions guarded by
# fast-math, so -O3 and -O0 produce the same bytes; the tests assert that
# against ik_llama's own library when a checkout is present.
set -e
HERE=${0:a:h}
OUT=${HERE}/libikq_iq1sr4.dylib
c++ -O3 -std=c++17 -fPIC -shared -o "${OUT}" "${HERE}/ikq_iq1sr4.cpp"
echo "built ${OUT}"

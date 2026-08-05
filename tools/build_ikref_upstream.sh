#!/bin/zsh
# Build the upstream cross-check against a CPU-only ik_llama build. The
# ik_llama checkout is read-and-build-only; nothing is written into that tree
# except its own build directory. Set IK_LLAMA_DIR to that checkout.
#
# This tool is optional. It proves the vendored codec reproduces ik_llama's
# own bytes; without it the vendored copy is still the reference the rest of
# the suite runs against, and the corresponding test skips.
set -e
HERE=${0:a:h}
if [[ -z ${IK_LLAMA_DIR:-} ]]; then
  echo "IK_LLAMA_DIR must be set to an ik_llama checkout" >&2
  exit 1
fi

IK=${IK_LLAMA_DIR}
BUILD=${IK}/build-cpu

if [[ ! -d ${IK} ]]; then
  echo "IK_LLAMA_DIR is not a directory: ${IK}" >&2
  exit 1
fi

if [[ ! -f ${BUILD}/ggml/src/libggml.dylib ]]; then
  cmake -B ${BUILD} -S ${IK} -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF \
        -DLLAMA_CURL=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
  cmake --build ${BUILD} --target ggml -j 12
fi

cc -O2 -std=c11 -o ${HERE}/ikref_upstream ${HERE}/ikref_upstream.c \
   -I${IK}/ggml/include \
   -L${BUILD}/ggml/src -lggml \
   -Wl,-rpath,${BUILD}/ggml/src -lm
echo "built ${HERE}/ikref_upstream"

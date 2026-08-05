"""IQ_K relayout kernels for MLX on Apple silicon.

Seven modules, in dependency order:

- :mod:`mlx_ikq.iq1grid` — the IQ1_S ternary grid codes and their decode
  value table, the one lookup the 1.5-bit member reconstructs from.
- :mod:`mlx_ikq.format` — the relayout wire definitions (routed members and
  dense higher-bpw members), the build-time pack from ik wire bytes, and the
  reference decodes.
- :mod:`mlx_ikq.codec` — the vendored ik encoders and the CPU dequantizers
  that are the bit-exactness references.
- :mod:`mlx_ikq.kernels` — the routed decode GEMV and stacked
  dequantization, ``mlx.fast.metal_kernel`` sources with fail-closed
  geometry guards.
- :mod:`mlx_ikq.dense_kernels` — the dense-tensor decode GEMV and
  dequantization for ``IQ4_KS``, ``IQ4_K``, ``IQ5_K``, ``IQ6_K``.
- :mod:`mlx_ikq.nn` — :class:`~mlx_ikq.nn.IkqSwitchLinear`, the stacked
  routed-expert switch module.
- :mod:`mlx_ikq.dense` — the module-level dense serving entry points.
"""

from mlx_ikq.dense import (
    dense_dequantized,
    dense_dequantized_range,
    dense_gemv,
    dense_linear,
    dense_value_table,
)
from mlx_ikq.format import (
    DENSE_MEMBERS,
    MEMBERS,
    bits_per_weight,
    component_shapes,
    decode,
    dense_component_shapes,
    pack,
)
from mlx_ikq.nn import IkqSwitchLinear

__all__ = [
    "DENSE_MEMBERS",
    "MEMBERS",
    "IkqSwitchLinear",
    "bits_per_weight",
    "component_shapes",
    "decode",
    "dense_component_shapes",
    "dense_dequantized",
    "dense_dequantized_range",
    "dense_gemv",
    "dense_linear",
    "dense_value_table",
    "pack",
]

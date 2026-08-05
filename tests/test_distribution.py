import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR_FILES = (
    "build.sh",
    "build_iq1sr4.sh",
    "ikq_dense.cpp",
    "ikq_dense.h",
    "ikq_iq1sr4.cpp",
    "ikq_iq1sr4.h",
    "ikq_iq2.cpp",
    "ikq_iq2.h",
)


def test_wheel_vendor_force_include_is_an_explicit_source_allowlist():
    with (ROOT / "pyproject.toml").open("rb") as file:
        config = tomllib.load(file)

    force_include = config["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    expected = {
        f"vendor/ik_llama/{name}": f"mlx_ikq_vendor/ik_llama/{name}"
        for name in VENDOR_FILES
    }

    assert force_include == expected
    assert all((ROOT / source).is_file() for source in expected)

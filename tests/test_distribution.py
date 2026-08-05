import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR_FILES = (
    "build.sh",
    "build_iq1sr4.sh",
    "iqk_dense.cpp",
    "iqk_dense.h",
    "iqk_iq1sr4.cpp",
    "iqk_iq1sr4.h",
    "iqk_iq2.cpp",
    "iqk_iq2.h",
)


def test_wheel_vendor_force_include_is_an_explicit_source_allowlist():
    with (ROOT / "pyproject.toml").open("rb") as file:
        config = tomllib.load(file)

    force_include = config["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    expected = {
        f"vendor/ik_llama/{name}": f"mlx_iqk_vendor/ik_llama/{name}"
        for name in VENDOR_FILES
    }

    assert force_include == expected
    assert all((ROOT / source).is_file() for source in expected)

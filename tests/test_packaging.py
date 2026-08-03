"""What the installed distribution has to carry, as opposed to what it computes.

A package can pass ``mypy --strict`` in its own repository and still hand out no
type information at all once it is installed. PEP 561 makes the marker file the
signal, and a type checker running against a consumer of this package ignores
every annotation in it unless that file is present inside the package directory.
Nothing else in the suite would notice its absence, which is why it is asserted
here rather than assumed.
"""

from __future__ import annotations

from pathlib import Path

import sensor_fusion

PACKAGE = Path(sensor_fusion.__file__).resolve().parent


def test_the_typing_marker_ships_inside_the_package() -> None:
    """PEP 561 requires ``py.typed`` in the package directory itself."""
    marker = PACKAGE / "py.typed"
    assert marker.is_file(), f"{marker} is absent, so annotations are invisible to consumers"


def test_the_typing_marker_is_empty() -> None:
    """The marker is a flag, not a manifest; content in it means something is confused."""
    assert (PACKAGE / "py.typed").read_bytes() == b""


def test_the_marker_sits_beside_the_package_init() -> None:
    """Guard the guard: the directory checked must be the importable package."""
    assert (PACKAGE / "__init__.py").is_file()
    assert PACKAGE.name == "sensor_fusion"

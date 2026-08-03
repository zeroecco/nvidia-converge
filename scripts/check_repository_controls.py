from __future__ import annotations

import sys
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import cast

_SOURCE_ROOT = str(Path(__file__).resolve(strict=True).parents[1])
if _SOURCE_ROOT not in sys.path:
    sys.path.insert(0, _SOURCE_ROOT)

main = cast(
    Callable[[list[str] | None], int],
    import_module("nvidia_converge.repository_controls").main,
)

if __name__ == "__main__":
    raise SystemExit(main(None))

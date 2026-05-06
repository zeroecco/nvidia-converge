from __future__ import annotations

import json
from importlib import resources
from typing import Literal

SchemaName = Literal["desired", "integration-results", "report"]


def load_schema(name: SchemaName) -> dict:
    with resources.files(__name__).joinpath(f"{name}.schema.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def schema_json(name: SchemaName) -> str:
    return json.dumps(load_schema(name), indent=2, sort_keys=True)

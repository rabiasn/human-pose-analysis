#!/usr/bin/env python3
"""Export the still-image feature table without changing any measurements.

The input JSON is produced by the feature extractor. Each image remains one
CSV row; null measurements become empty fields, numeric precision is retained,
and quality flags are copied literally. Existing output files are never replaced.
CSV does not store types, so numeric_columns in the source JSON is its schema.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def encoded(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def validate_table(data: dict) -> tuple[list[str], list[list]]:
    columns = data["columns"]
    rows = data["rows"]
    if not columns or any(not isinstance(c, str) or not c for c in columns):
        raise ValueError("Columns must be nonempty strings.")
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate column names.")
    numeric = set(data["numeric_columns"])
    if not numeric <= set(columns):
        raise ValueError("Numeric schema contains unknown columns.")
    if not set(data["feature_columns"]) <= numeric:
        raise ValueError("Feature columns must be declared numeric.")
    if not isinstance(rows, list):
        raise ValueError("Rows must be a list.")
    for row_number, row in enumerate(rows, 1):
        if not isinstance(row, list) or len(row) != len(columns):
            raise ValueError(f"Row {row_number} has an invalid width.")
        for name, value in zip(columns, row):
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"Unsupported value in row {row_number}, {name}.")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Nonfinite value in row {row_number}, {name}.")
            if name in numeric and value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"Nonnumeric value in row {row_number}, {name}.")
    return columns, rows


def verify_roundtrip(path: Path, columns: list[str], rows: list[list]) -> dict:
    with path.open(encoding="utf-8", newline="") as handle:
        saved = list(csv.reader(handle))
    if saved[0] != columns or len(saved) != len(rows) + 1:
        raise AssertionError("CSV header or row count changed during export.")
    numeric_count = 0
    null_count = 0
    boolean_count = 0
    for row_number, (original, restored) in enumerate(zip(rows, saved[1:]), 1):
        if len(restored) != len(columns):
            raise AssertionError(f"CSV width mismatch in row {row_number}.")
        for name, value, cell in zip(columns, original, restored):
            if value is None:
                null_count += 1
                same = cell == ""
            elif isinstance(value, bool):
                boolean_count += 1
                same = cell == encoded(value)
            elif isinstance(value, int):
                numeric_count += 1
                same = int(cell) == value
            elif isinstance(value, float):
                numeric_count += 1
                same = float(cell).hex() == value.hex()
            else:
                same = cell == value
            if not same:
                raise AssertionError(f"CSV value mismatch in row {row_number}, {name}.")
    return {
        "rows": len(rows),
        "columns": len(columns),
        "numeric_values_roundtripped_exactly": numeric_count,
        "null_values_preserved_as_empty": null_count,
        "boolean_values_preserved_as_true_false": boolean_count,
        "all_text_values_preserved": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    with args.input_json.open(encoding="utf-8") as handle:
        data = json.load(handle)
    columns, rows = validate_table(data)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation deliberately refuses to replace a previous dataset.
    with args.output_csv.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows([[encoded(value) for value in row] for row in rows])
    result = verify_roundtrip(args.output_csv, columns, rows)
    result["feature_columns"] = len(data["feature_columns"])
    result["csv_sha256"] = hashlib.sha256(args.output_csv.read_bytes()).hexdigest()
    result["output_csv"] = str(args.output_csv.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

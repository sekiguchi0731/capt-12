from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

DAY_PATTERN = re.compile(r"(?:^|/)day_int=([^/]+)(?:/|$)")


@dataclass
class ColumnInspection:
    name: str
    arrow_type: str
    metadata_null_count: int | None
    sample_null_count: int
    sample_cardinality: int | None


@dataclass
class DataInspection:
    data_root: str
    files: int
    days: list[int]
    rows: int
    row_groups: int
    columns: list[ColumnInspection]
    rows_by_day: dict[int, int]
    schemas_identical: bool
    sample_rows: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_parquet(data_root: str | Path) -> list[Path]:
    root = Path(data_root)
    if root.is_file() and root.suffix == ".parquet":
        return [root]
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet files found under {root}")
    return files


def day_from_path(path: str | Path) -> int:
    match = DAY_PATTERN.search(Path(path).as_posix())
    if not match:
        raise ValueError(f"cannot infer day_int partition from {path}")
    return int(match.group(1))


def inspect_parquet(data_root: str | Path, sample_rows_per_file: int = 2048) -> DataInspection:
    files = discover_parquet(data_root)
    schema: pa.Schema | None = None
    schemas_identical = True
    total_rows = 0
    row_groups = 0
    rows_by_day: dict[int, int] = {}
    null_counts: dict[str, int] = {}
    stats_present: dict[str, bool] = {}
    samples: list[pa.Table] = []
    for path in files:
        parquet = pq.ParquetFile(path)
        this_schema = parquet.schema_arrow
        if schema is None:
            schema = this_schema
        elif not schema.equals(this_schema):
            schemas_identical = False
        total_rows += parquet.metadata.num_rows
        row_groups += parquet.metadata.num_row_groups
        day = day_from_path(path)
        rows_by_day[day] = rows_by_day.get(day, 0) + parquet.metadata.num_rows
        for rg_idx in range(parquet.metadata.num_row_groups):
            rg = parquet.metadata.row_group(rg_idx)
            for col_idx, name in enumerate(this_schema.names):
                stat = rg.column(col_idx).statistics
                if stat is not None and stat.has_null_count:
                    null_counts[name] = null_counts.get(name, 0) + stat.null_count
                    stats_present[name] = True
        if sample_rows_per_file > 0:
            try:
                batch = next(parquet.iter_batches(batch_size=sample_rows_per_file))
                samples.append(pa.Table.from_batches([batch]))
            except StopIteration:
                pass
    assert schema is not None
    sample = pa.concat_tables(samples, promote_options="default") if samples else pa.table({})
    columns: list[ColumnInspection] = []
    for field in schema:
        array = sample[field.name] if sample.num_columns else None
        cardinality = None
        if array is not None and not (pa.types.is_list(array.type) or pa.types.is_large_list(array.type)):
            try:
                cardinality = len(array.combine_chunks().unique())
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                cardinality = None
        columns.append(
            ColumnInspection(
                name=field.name,
                arrow_type=str(field.type),
                metadata_null_count=null_counts.get(field.name) if stats_present.get(field.name) else None,
                sample_null_count=array.null_count if array is not None else 0,
                sample_cardinality=cardinality,
            )
        )
    return DataInspection(
        data_root=str(Path(data_root)),
        files=len(files),
        days=sorted(rows_by_day),
        rows=total_rows,
        row_groups=row_groups,
        columns=columns,
        rows_by_day=rows_by_day,
        schemas_identical=schemas_identical,
        sample_rows=sample.num_rows,
    )


def write_inspection_json(inspection: DataInspection, path: str | Path) -> None:
    Path(path).write_text(json.dumps(inspection.to_dict(), indent=2, sort_keys=True) + "\n")


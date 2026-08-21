"""Local Spark / Databricks feature-engineering engine.

Two back-ends, both free and offline-capable:

* ``pyspark`` (local mode) — runs on the local machine, no cluster needed.
  Install: ``pip install pyspark``.
* ``databricks-sdk`` — connects to a Databricks workspace ONLY when the
  environment variables ``DATABRICKS_HOST`` and ``DATABRICKS_TOKEN`` are set.
  Without them (the default) the engine falls back to local pyspark or, if
  pyspark is missing, to a pure-``pandas``/``polars`` reader so the module
  always imports.

Typical use (feature engineering over Silverwing JSONL corpora)::

    from foundation.ops.spark_engine import spark_session, df_from_jsonl
    spark = spark_session(app_name="silverwing-feat")
    df = df_from_jsonl(spark, "datasets/raw/*.jsonl")
    feats = compute_text_features(df, text_col="text")
    feats.write.parquet("experiments/features/text_feats/")

When neither Spark nor Databricks are available, ``df_from_jsonl`` returns a
lightweight Pandas-backed frame so downstream code keeps working.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .._compat import optional_dependency

pyspark = optional_dependency("pyspark")
databricks_sdk = optional_dependency("databricks.sdk")

_LOCAL_MASTER = "local[*]"


def _spark_available() -> bool:
    try:
        import pyspark  # noqa: F401
        from pyspark.sql import SparkSession  # noqa: F401

        return True
    except Exception:
        return False


def _databricks_configured() -> bool:
    return bool(os.environ.get("DATABRICKS_HOST")) and bool(os.environ.get("DATABRICKS_TOKEN"))


def backend() -> str:
    """Return the active backend name."""
    if _spark_available():
        return "pyspark"
    if _databricks_configured():
        return "databricks"
    return "polars"  # last-resort fallback for feature transforms


def spark_session(app_name: str = "silverwing", master: str = _LOCAL_MASTER, **kwargs: Any):
    """Create a local SparkSession. Raises if pyspark is not installed."""
    if not _spark_available():
        raise RuntimeError("pyspark is not installed; install with `pip install pyspark`")
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", kwargs.get("shuffle_partitions", "4"))
        .config("spark.driver.memory", kwargs.get("driver_memory", "2g"))
        .getOrCreate()
    )


def databricks_client() -> Any:
    """Return a configured Databricks SDK client (requires env vars)."""
    if not _databricks_configured():
        raise RuntimeError(
            "Set DATABRICKS_HOST and DATABRICKS_TOKEN to use the Databricks backend"
        )
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient()


def df_from_jsonl(spark_or_session, path_glob: str | Path) -> Any:
    """Load JSONL into a Spark DataFrame (or pandas fallback)."""
    path_str = str(path_glob)
    if spark_or_session is None or not _spark_available():
        import json

        try:
            import polars as pl
        except Exception:
            import pandas as pd

            pl = pd  # type: ignore
        records = []
        for p in sorted(Path(".").glob(path_str)):
            if not p.is_file():
                continue
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if records and hasattr(pl, "DataFrame"):
            return pl.DataFrame(records)
        return records

    spark = spark_or_session
    return spark.read.json(path_str)


def compute_text_features(df, text_col: str = "text", id_col: str = "id") -> Any:
    """Compute basic NLP features. Works on Spark DF or pandas/polars DF."""
    if _spark_available() and hasattr(df, "select"):
        from pyspark.sql import functions as F

        return df.select(
            id_col,
            text_col,
            F.length(F.col(text_col)).alias("char_len"),
            F.size(F.split(F.col(text_col), "\\s+")).alias("word_count"),
            F.lower(F.col(text_col)).alias("text_lower"),
        )

    # pandas / polars fallback
    try:
        import polars as pl
    except Exception:

        return _pandas_text_features(df, text_col, id_col)

    if hasattr(df, "select"):
        return df.with_columns(
            pl.col(text_col).str.len_chars().alias("char_len"),
            pl.col(text_col).str.n_words().alias("word_count"),
            pl.col(text_col).str.to_lowercase().alias("text_lower"),
        )
    return df


def _pandas_text_features(df, text_col: str, id_col: str):
    import pandas as pd

    if isinstance(df, list):
        df = pd.DataFrame(df)
    out = df[[id_col, text_col]].copy() if id_col in df.columns else df[[text_col]].copy()
    out["char_len"] = out[text_col].astype(str).str.len()
    out["word_count"] = out[text_col].astype(str).str.split().str.len()
    out["text_lower"] = out[text_col].astype(str).str.lower()
    return out


def write_parquet(df, path: str | Path, mode: str = "overwrite") -> str:
    """Persist a DataFrame to parquet (Spark or pandas/polars fallback)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if _spark_available() and hasattr(df, "write"):
        df.write.mode(mode).parquet(str(out))
        return str(out)
    try:
        import polars as pl

        if isinstance(df, pl.DataFrame):
            df.write_parquet(str(out))
            return str(out)
    except Exception:
        pass
    if hasattr(df, "to_parquet"):
        df.to_parquet(str(out))
        return str(out)
    raise TypeError("unsupported DataFrame type for parquet write")


def tracker() -> dict[str, Any]:
    """Introspect the current Spark/Databricks availability."""
    return {
        "backend": backend(),
        "pyspark": _spark_available(),
        "databricks": _databricks_configured(),
    }

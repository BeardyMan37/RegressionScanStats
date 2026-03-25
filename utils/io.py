from pathlib import Path
import pandas as pd


def read_table(path: Path | str) -> pd.DataFrame:
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    sp = str(path).lower()
    if sp.endswith((".parquet", ".pq")):
        try:
            return pd.read_parquet(path)
        except Exception as e:
            raise RuntimeError(
                f"Reading parquet requires pyarrow or fastparquet. Error: {e}"
            )
    elif sp.endswith((".csv", ".txt")):
        return pd.read_csv(path)
    else:
        # try parquet then csv
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path)

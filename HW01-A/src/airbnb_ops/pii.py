import hashlib

import pandas as pd


DIRECT_PII_COLUMNS = ["host_name"]


def pseudonymize_value(value, salt: str = "qbc12") -> str:
    raw = f"{salt}:{value}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def handle_pii(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    for column in DIRECT_PII_COLUMNS:
        if column in cleaned.columns:
            cleaned = cleaned.drop(columns=[column])

    if "host_id" in cleaned.columns:
        cleaned["host_key"] = cleaned["host_id"].apply(pseudonymize_value)
        cleaned = cleaned.drop(columns=["host_id"])

    return cleaned

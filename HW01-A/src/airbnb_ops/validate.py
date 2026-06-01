import pandas as pd


REQUIRED_OUTPUT_COLUMNS = {
    "neighbourhood",
    "num_listings",
    "avg_price",
    "median_price",
    "avg_minimum_nights",
    "availability_365_avg",
    "total_reviews",
    "reviews_per_listing",
    "tourism_segment",
    "priority_level",
}

PII_COLUMNS = {
    "host_name",
    "host_id",
    "reviewer_name",
    "reviewer_id",
    "listing_url",
    "host_url",
}


def validate_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        raise ValueError("Summary output is empty.")

    missing = REQUIRED_OUTPUT_COLUMNS - set(summary.columns)
    if missing:
        raise ValueError(f"Summary is missing required columns: {sorted(missing)}")

    pii_leaks = PII_COLUMNS & set(summary.columns)
    if pii_leaks:
        raise ValueError(f"PII columns leaked into output: {sorted(pii_leaks)}")

    if summary["neighbourhood"].isna().any():
        raise ValueError("neighbourhood contains null values.")

    if (summary["num_listings"] <= 0).any():
        raise ValueError("num_listings must be greater than 0.")

    if (summary["avg_price"] < 0).any():
        raise ValueError("avg_price must be non-negative.")

    if not summary["availability_365_avg"].between(0, 365).all():
        raise ValueError("availability_365_avg must be between 0 and 365.")

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text


try:
    from airflow.decorators import dag, task
    from airflow.models import Variable
except ImportError:
    from airflow.sdk import dag, task, Variable


DAG_ID = "qbc12_hw01_alirezaabavi_airbnb_pipeline"
STUDENT_SCHEMA = "student_alirezaabavi"
SUMMARY_VIEW = f"{STUDENT_SCHEMA}.airbnb_neighbourhood_summary"


def make_engine():
    """
    Create a PostgreSQL connection from Airflow Variables.

    Important:
    No username, password, host, database URL, or private credential is hard-coded.
    """

    host = Variable.get("QBC12_DB_HOST")
    port = Variable.get("QBC12_DB_PORT", default_var="5432")
    database = Variable.get("QBC12_DB_NAME")
    username = Variable.get("QBC12_DB_USER")
    password = Variable.get("QBC12_DB_PASSWORD")

    url = (
        "postgresql+psycopg2://"
        f"{quote_plus(username)}:{quote_plus(password)}"
        f"@{host}:{port}/{database}"
    )

    return create_engine(url, pool_pre_ping=True)


@dag(
    dag_id=DAG_ID,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["qbc12", "hw01", "airbnb"],
)
def airbnb_pipeline():

    @task
    def read_config() -> dict:
        """
        Read non-secret pipeline configuration.

        The source table names are configurable through Airflow Variables.
        The default names are placeholders and can be overridden in Airflow.
        """

        return {
            "student_schema": STUDENT_SCHEMA,
            "summary_view": SUMMARY_VIEW,
            "source_listings_table": Variable.get(
                "QBC12_AIRBNB_LISTINGS_TABLE",
                default_var="public.listings_sample",
            ),
            "source_segments_table": Variable.get(
                "QBC12_AIRBNB_SEGMENTS_TABLE",
                default_var="public.neighbourhood_segments",
            ),
        }

    @task
    def refresh_summary(config: dict) -> dict:
        """
        Recreate the Airbnb neighbourhood summary materialized view.
        """

        engine = make_engine()

        student_schema = config["student_schema"]
        summary_view = config["summary_view"]
        listings_table = config["source_listings_table"]
        segments_table = config["source_segments_table"]

        refresh_sql = f"""
        CREATE SCHEMA IF NOT EXISTS {student_schema};

        DROP MATERIALIZED VIEW IF EXISTS {summary_view};

        CREATE MATERIALIZED VIEW {summary_view} AS
        WITH neighbourhood_summary AS (
            SELECT
                neighbourhood,
                COUNT(listing_id) AS num_listings,
                AVG(price)::numeric(12, 2) AS avg_price,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price)::numeric(12, 2) AS median_price,
                AVG(minimum_nights)::numeric(12, 2) AS avg_minimum_nights,
                AVG(availability_365)::numeric(12, 2) AS availability_365_avg,
                SUM(number_of_reviews) AS total_reviews,
                (
                    SUM(number_of_reviews)::numeric
                    / NULLIF(COUNT(listing_id), 0)
                )::numeric(12, 2) AS reviews_per_listing
            FROM {listings_table}
            GROUP BY neighbourhood
        )
        SELECT
            ns.neighbourhood,
            ns.num_listings,
            ns.avg_price,
            ns.median_price,
            ns.avg_minimum_nights,
            ns.availability_365_avg,
            ns.total_reviews,
            ns.reviews_per_listing,
            COALESCE(seg.tourism_segment, 'unknown') AS tourism_segment,
            COALESCE(seg.priority_level, 'unknown') AS priority_level
        FROM neighbourhood_summary ns
        LEFT JOIN {segments_table} seg
            ON ns.neighbourhood = seg.neighbourhood
        ORDER BY ns.neighbourhood;

        CREATE INDEX IF NOT EXISTS idx_airbnb_neighbourhood_summary_neighbourhood
        ON {summary_view} (neighbourhood);

        CREATE INDEX IF NOT EXISTS idx_airbnb_neighbourhood_summary_priority
        ON {summary_view} (priority_level);
        """

        with engine.begin() as conn:
            conn.execute(text(refresh_sql))

        return {
            "status": "refreshed",
            "refreshed_object": summary_view,
        }

    @task
    def validate_summary(config: dict, refresh_result: dict) -> dict:
        """
        Validate the materialized view.

        Required HW01-C checks:
        - row_count > 0
        - null_neighbourhoods == 0
        - bad_prices == 0
        - bad_availability == 0
        """

        engine = make_engine()
        summary_view = config["summary_view"]

        validation_sql = f"""
        SELECT
            COUNT(*) AS row_count,

            COUNT(*) FILTER (
                WHERE neighbourhood IS NULL
            ) AS null_neighbourhoods,

            COUNT(*) FILTER (
                WHERE avg_price IS NULL
                   OR avg_price < 0
                   OR median_price IS NULL
                   OR median_price < 0
            ) AS bad_prices,

            COUNT(*) FILTER (
                WHERE availability_365_avg IS NULL
                   OR availability_365_avg < 0
                   OR availability_365_avg > 365
            ) AS bad_availability
        FROM {summary_view};
        """

        with engine.begin() as conn:
            row = conn.execute(text(validation_sql)).mappings().one()

        result = dict(row)

        result["passed"] = (
            result["row_count"] > 0
            and result["null_neighbourhoods"] == 0
            and result["bad_prices"] == 0
            and result["bad_availability"] == 0
        )

        result["refreshed_object"] = refresh_result["refreshed_object"]

        return result

    @task.branch
    def choose_report_path(validation_result: dict) -> str:
        if validation_result["passed"]:
            return "write_success_report"

        return "write_failure_report"

    @task
    def write_success_report(config: dict, validation_result: dict) -> str:
        report_path = Path("/tmp/hw01_c_airflow_success.md")

        report = f"""# HW01-C Airflow Run Report

Status: SUCCESS

DAG ID: {DAG_ID}

Refreshed object: {validation_result["refreshed_object"]}

Validation result:

- row_count: {validation_result["row_count"]}
- null_neighbourhoods: {validation_result["null_neighbourhoods"]}
- bad_prices: {validation_result["bad_prices"]}
- bad_availability: {validation_result["bad_availability"]}
- passed: {validation_result["passed"]}

Credentials are loaded from Airflow Variables.
No credentials are hard-coded in the DAG.
"""

        report_path.write_text(report, encoding="utf-8")
        return str(report_path)

    @task
    def write_failure_report(config: dict, validation_result: dict) -> str:
        report_path = Path("/tmp/hw01_c_airflow_failure.md")

        report = f"""# HW01-C Airflow Run Report

Status: FAILURE

DAG ID: {DAG_ID}

Refreshed object: {validation_result["refreshed_object"]}

Validation result:

- row_count: {validation_result["row_count"]}
- null_neighbourhoods: {validation_result["null_neighbourhoods"]}
- bad_prices: {validation_result["bad_prices"]}
- bad_availability: {validation_result["bad_availability"]}
- passed: {validation_result["passed"]}

The DAG failed because one or more validation checks did not pass.
"""

        report_path.write_text(report, encoding="utf-8")

        raise ValueError(f"Validation failed: {validation_result}")

    config = read_config()
    refresh_result = refresh_summary(config)
    validation_result = validate_summary(config, refresh_result)

    branch = choose_report_path(validation_result)

    success_report = write_success_report(config, validation_result)
    failure_report = write_failure_report(config, validation_result)

    branch >> [success_report, failure_report]


airbnb_pipeline()

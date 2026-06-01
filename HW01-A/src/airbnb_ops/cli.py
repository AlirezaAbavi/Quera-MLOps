from pathlib import Path

import typer
from rich.console import Console

from airbnb_ops.config import PipelineConfig
from airbnb_ops.extract import read_csv_checked
from airbnb_ops.pii import handle_pii
from airbnb_ops.transform import build_neighbourhood_summary
from airbnb_ops.validate import validate_summary

app = typer.Typer(help="Airbnb operations data pipeline.")
console = Console()

@app.callback()
def main() -> None:
    """Airbnb operations data pipeline."""
    pass


def write_report(summary, report_path: Path, output_path: Path) -> None:
    """Write a short markdown run report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = f"""# HW01-A Run Report

## Pipeline status

Success.

## Output

- Output CSV: `{output_path}`
- Number of neighbourhoods: {len(summary)}
- Total listings: {int(summary["num_listings"].sum())}

## Validation checks

- Output is not empty.
- Required output columns exist.
- PII columns are not present.
- `neighbourhood` has no null values.
- `num_listings` is greater than 0.
- `avg_price` is non-negative.
- `availability_365_avg` is between 0 and 365.

## Columns

{", ".join(summary.columns)}
"""

    report_path.write_text(report, encoding="utf-8")


@app.command()
def run() -> None:
    """Run the Airbnb neighbourhood summary pipeline."""
    config = PipelineConfig()

    console.print("[bold blue]Reading raw inputs...[/bold blue]")
    listings = read_csv_checked(config.listings_path)
    segments = read_csv_checked(config.segments_path)

    console.print("[bold blue]Handling PII...[/bold blue]")
    clean_listings = handle_pii(listings)

    console.print("[bold blue]Building neighbourhood summary...[/bold blue]")
    summary = build_neighbourhood_summary(clean_listings, segments)

    console.print("[bold blue]Validating summary...[/bold blue]")
    validate_summary(summary)

    console.print("[bold blue]Writing outputs...[/bold blue]")
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.output_path, index=False)
    write_report(summary, config.report_path, config.output_path)

    console.print(f"[bold green]Done.[/bold green] Wrote {config.output_path}")
    console.print(f"[bold green]Done.[/bold green] Wrote {config.report_path}")


if __name__ == "__main__":
    app()

# HW01-A Run Report

## Pipeline status

Success.

## Output

- Output CSV: `data/processed/airbnb_neighbourhood_summary.csv`
- Number of neighbourhoods: 5
- Total listings: 8

## Validation checks

- Output is not empty.
- Required output columns exist.
- PII columns are not present.
- `neighbourhood` has no null values.
- `num_listings` is greater than 0.
- `avg_price` is non-negative.
- `availability_365_avg` is between 0 and 365.

## Columns

neighbourhood, num_listings, avg_price, median_price, avg_minimum_nights, availability_365_avg, total_reviews, reviews_per_listing, tourism_segment, priority_level

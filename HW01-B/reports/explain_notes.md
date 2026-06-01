# EXPLAIN ANALYZE Notes

## Observation 1 — The final result is tiny, but the query does large intermediate work

The final query returns only 22 neighbourhood rows.

Evidence from the plan:

- Final `Sort` node: `actual time=349.881..350.014 rows=22`
- Final `GroupAggregate` node: `actual time=343.939..349.984 rows=22`
- Group key: `l.neighbourhood_id`

This means the output is small, but PostgreSQL still has to process much larger intermediate datasets before producing the final neighbourhood summary.

## Observation 2 — Calendar aggregation is the largest expensive part

The query filters the latest 30 days from `core.calendar_day` using the `idx_calendar_date` index.

Evidence from the plan:

- `Bitmap Index Scan on idx_calendar_date`
- `Bitmap Heap Scan on calendar_day c`
- Calendar rows processed: `rows=314400`
- Calendar heap blocks read: `Heap Blocks: exact=12416`
- Calendar scan time: `actual time=44.902..106.749`
- Calendar aggregation by listing: `HashAggregate`
- Calendar aggregation time: `actual time=234.317..243.188`
- Calendar aggregation memory: `Memory Usage: 4241kB`

So the index helps, but the query still reads and aggregates 314,400 calendar rows before joining to listings.

## Observation 3 — Review aggregation also adds noticeable work

The query aggregates `core.review` by `listing_id` before joining the result to listings.

Evidence from the plan:

- Review aggregation node: `Finalize HashAggregate`
- Group key: `review.listing_id`
- Review aggregate rows produced: `rows=9383`
- Review aggregation time: `actual time=73.248..74.929`
- Review buffers: `shared hit=480 read=7736`

This is smaller than the calendar work, but it is still repeated every time the baseline query runs.

## Baseline timing summary

Python-side timings from repeated runs:

- Run 1: `0.6216` seconds
- Run 2: `0.6002` seconds
- Run 3: `0.6033` seconds

Best Python-side runtime: `0.6002` seconds  
Average Python-side runtime: `0.6084` seconds

The database-side plan finishes around 350 ms, while the Python-side timing is around 0.60 seconds because it includes client overhead and loading the result into pandas.

## Optimization decision

The baseline query is correct, but it repeats calendar aggregation, review aggregation, joins, and final grouping every time it runs.

A materialized view is a good optimization because it stores the prepared neighbourhood-level result. Future reads can query the materialized view directly instead of recalculating the full baseline query.

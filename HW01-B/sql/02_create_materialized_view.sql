
drop materialized view if exists "student_alireza_abouei".mv_airbnb_neighbourhood_summary;

create materialized view "student_alireza_abouei".mv_airbnb_neighbourhood_summary as
with max_calendar_date as (
    select max(date) as max_date
    from core.calendar_day
),
calendar_30 as (
    select
        c.listing_id,
        avg(coalesce(c.adjusted_price, c.price)) as avg_calendar_price_30,
        avg(case when c.available then 1.0 else 0.0 end) as availability_30_rate
    from core.calendar_day c
    cross join max_calendar_date m
    where c.date >= m.max_date - interval '29 days'
      and c.date <= m.max_date
    group by c.listing_id
),
review_counts as (
    select
        listing_id,
        count(*) as total_reviews
    from core.review
    group by listing_id
)
select
    l.neighbourhood_id::text as neighbourhood,
    count(*) as num_listings,
    round(avg(l.listing_price), 2) as avg_price,
    percentile_cont(0.5) within group (order by l.listing_price) as median_price,
    round(avg(l.minimum_nights), 2) as avg_minimum_nights,
    coalesce(sum(r.total_reviews), 0) as total_reviews,
    round(
        coalesce(sum(r.total_reviews), 0)::numeric / nullif(count(*), 0),
        2
    ) as reviews_per_listing,
    round(avg(c.availability_30_rate), 4) as availability_30_rate
from core.listing l
left join calendar_30 c
    on l.listing_id = c.listing_id
left join review_counts r
    on l.listing_id = r.listing_id
group by l.neighbourhood_id;

create index idx_mv_airbnb_neighbourhood_summary_neighbourhood
on "student_alireza_abouei".mv_airbnb_neighbourhood_summary (neighbourhood);

create index idx_mv_airbnb_neighbourhood_summary_num_listings
on "student_alireza_abouei".mv_airbnb_neighbourhood_summary (num_listings desc);

create index idx_mv_airbnb_neighbourhood_summary_avg_price
on "student_alireza_abouei".mv_airbnb_neighbourhood_summary (avg_price);

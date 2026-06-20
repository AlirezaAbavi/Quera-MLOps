from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ListingFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_type: str
    property_type: str
    neighbourhood_name: str

    accommodates: int = Field(ge=0)
    bedrooms: Optional[float] = Field(default=None, ge=0)
    beds: Optional[float] = Field(default=None, ge=0)
    bathrooms: Optional[float] = Field(default=None, ge=0)

    listing_price: float = Field(ge=0)
    minimum_nights: int = Field(ge=0)
    maximum_nights: int = Field(ge=0)

    instant_bookable: bool
    host_is_superhost: bool

    host_listing_count: int = Field(ge=0)

    total_reviews_before_cutoff: int = Field(ge=0)
    unique_reviewers_before_cutoff: int = Field(ge=0)

    avg_comment_len_before_cutoff: Optional[float] = Field(default=None, ge=0)
    max_comment_len_before_cutoff: Optional[float] = Field(default=None, ge=0)
    days_since_last_review: Optional[float] = Field(default=None, ge=0)

    available_days_last_90d: int = Field(ge=0)
    available_rate_last_90d: float = Field(ge=0, le=1)
    avg_minimum_nights_calendar_last_90d: Optional[float] = Field(default=None, ge=0)
    avg_maximum_nights_calendar_last_90d: Optional[float] = Field(default=None, ge=0)

    available_days_last_30d: int = Field(ge=0)
    available_rate_last_30d: float = Field(ge=0, le=1)
    avg_minimum_nights_calendar_last_30d: Optional[float] = Field(default=None, ge=0)
    avg_maximum_nights_calendar_last_30d: Optional[float] = Field(default=None, ge=0)


class PredictionResponse(BaseModel):
    listing_id: Optional[int] = None
    prediction: int = Field(ge=0, le=1)
    probability_high_demand: float = Field(ge=0, le=1)
    model_run_id: str

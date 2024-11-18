from typing import Annotated
from pydantic import BaseModel, Field, field_validator


class FeatureDataSchema(BaseModel):
    X: Annotated[
        int,
        Field(
            ge=1,
            le=9,
            description="X-axis spatial coordinate. Must be an integer between 1 and 9",
        ),
    ]
    Y: Annotated[
        int,
        Field(
            ge=2,
            le=9,
            description="Y-axis spatial coordinate. Must be an integer between 2 and 9",
        ),
    ]
    month: Annotated[
        str,
        Field(
            pattern="^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)$",
            description="Month of the year. Must be one of 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', or 'dec'",
        ),
    ]
    day: Annotated[
        str,
        Field(
            pattern="^(mon|tue|wed|thu|fri|sat|sun)$",
            description="Day of the week. Must be one of 'mon', 'tue', 'wed', 'thu', 'fri', 'sat', or 'sun'",
        ),
    ]
    DMC: Annotated[
        float,
        Field(
            ge=1.1,
            le=291.3,
            description="Duff Moisture Code Index from the FWI system indicating the moisture content of deep organic layers. Must be a float between 1.1 and 291.3",
        ),
    ]
    FFMC: Annotated[
        float,
        Field(
            ge=18.7,
            le=96.20,
            description="Fine Fuel Moisture Code Index from the FWI system indicating the moisture content of litter and other cured fine fuels. Must a float between 18.7 and 96.20",
        ),
    ]
    DC: Annotated[
        float,
        Field(
            ge=7.9,
            le=860.6,
            description="Drought Code Index from the FWI system indicating the moisture content of deep compact organic layers. Must be a float between 7.9 and 860.6",
        ),
    ]
    ISI: Annotated[
        float,
        Field(
            ge=0.0,
            le=56.10,
            description="Initial Spread Index from the FWI system which combines the effects of wind and FFMC to estimate the rate of fire spread. Must be a float between 0.0 and 56.10",
        ),
    ]
    temp: Annotated[
        float,
        Field(
            ge=2.2,
            le=33.30,
            description="Temperature in ˚C recorded at noon (standard time). Must be a float between 2.2 and 33.30",
        ),
    ]
    RH: Annotated[
        int,
        Field(
            ge=15,
            le=100,
            description="Relative humidity in '%' recorded at noon (standard time). Must be an integer between 15 and 100",
        ),
    ]
    wind: Annotated[
        float,
        Field(
            ge=0.40,
            le=9.40,
            description="Wind speed in km/h recorded at noon (standard time). Must be a float between 0.40 and 9.40",
        ),
    ]
    rain: Annotated[
        float,
        Field(
            ge=0.0,
            le=6.4,
            description="Outside rain in mm/m recorded at noon (standard time). Must be a float between 0.0 and 6.4",
        ),
    ]

    @field_validator("*", mode="before")
    def check_type_and_range(cls, value, field):
        if isinstance(value, int) and field.annotation == float:
            raise ValueError(f"{field.name} must be a float.")
        if isinstance(value, float) and field.annotation == int:
            raise ValueError(f"{field.name} must be an integer.")
        return value


class InputData(BaseModel):
    features: FeatureDataSchema

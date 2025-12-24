"""
Truck Delay Feature Engineering Pipeline
Author: ND Fyneface
Description:
- Extracts data from PostgreSQL & MySQL
- Performs feature engineering
- Loads data into Hopsworks Feature Store
"""

import os
import logging
from typing import Dict

import pandas as pd
import numpy as np
import psycopg2
import pymysql
import hopsworks


# -------------------------------------------------------------------
# LOGGING CONFIG
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# DATABASE CONFIG (ENV VARS ONLY)
# -------------------------------------------------------------------
POSTGRES_CONFIG: Dict[str, str] = {
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": os.getenv("POSTGRES_HOST"),
    "database": os.getenv("POSTGRES_DB"),
    "port": "5432"
}

MYSQL_CONFIG: Dict[str, str] = {
    "host": os.getenv("MYSQL_HOST"),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DB"),
}


# -------------------------------------------------------------------
# DATABASE READERS
# -------------------------------------------------------------------
def read_postgres(query: str) -> pd.DataFrame:
    logger.info("Reading data from PostgreSQL")
    with psycopg2.connect(**POSTGRES_CONFIG) as conn:
        return pd.read_sql(query, conn)


def read_mysql(query: str) -> pd.DataFrame:
    logger.info("Reading data from MySQL")
    with pymysql.connect(**MYSQL_CONFIG) as conn:
        return pd.read_sql(query, conn)


# -------------------------------------------------------------------
# TRANSFORMATIONS
# -------------------------------------------------------------------
def preprocess_drivers(df: pd.DataFrame) -> pd.DataFrame:
    df["gender"] = df["gender"].fillna("Unknown")
    df["driving_style"] = df["driving_style"].fillna("Unknown")
    df["event_time"] = pd.to_datetime("2023-08-23")
    return df.sort_values(["event_time", "driver_id"])


def preprocess_trucks(df: pd.DataFrame) -> pd.DataFrame:
    df["fuel_type"] = df["fuel_type"].replace("", "Unknown")
    df["event_time"] = pd.to_datetime("2023-08-23")
    return df.sort_values(["event_time", "truck_id"])


def preprocess_routes(df: pd.DataFrame) -> pd.DataFrame:
    df["event_time"] = pd.to_datetime("2023-08-23")
    return df.sort_values(["event_time", "route_id"])


def preprocess_weather(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["city_id", "date", "hour"])
    df = df.drop(
        columns=[
            "chanceofrain",
            "chanceoffog",
            "chanceofsnow",
            "chanceofthunder"
        ],
        errors="ignore"
    )
    return df.sort_values(["date", "city_id", "hour"])


# -------------------------------------------------------------------
# FEATURE STORE
# -------------------------------------------------------------------
def get_feature_store():
    logger.info("Logging into Hopsworks")
    project = hopsworks.login()
    return project.get_feature_store()


def create_feature_group(
    fs,
    name: str,
    version: int,
    df: pd.DataFrame,
    primary_key: list,
    event_time: str,
    online_enabled: bool
):
    fg = fs.get_or_create_feature_group(
        name=name,
        version=version,
        primary_key=primary_key,
        event_time=event_time,
        online_enabled=online_enabled,
        description=f"{name} feature group"
    )
    fg.insert(df)
    fg.statistics_config = {
        "enabled": True,
        "histograms": True,
        "correlations": True
    }
    fg.update_statistics_config()
    fg.compute_statistics()
    logger.info(f"Feature group {name} loaded successfully")


# -------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------
def main():
    logger.info("🚀 Starting Truck Delay Feature Pipeline")

    # -----------------------------
    # EXTRACT
    # -----------------------------
    drivers_df = read_mysql("SELECT * FROM drivers_details")
    trucks_df = read_mysql("SELECT * FROM truck_details")
    routes_df = read_postgres("SELECT * FROM routes_details")
    weather_df = read_mysql("SELECT * FROM city_weather")

    # -----------------------------
    # TRANSFORM
    # -----------------------------
    drivers_df = preprocess_drivers(drivers_df)
    trucks_df = preprocess_trucks(trucks_df)
    routes_df = preprocess_routes(routes_df)
    weather_df = preprocess_weather(weather_df)

    # -----------------------------
    # LOAD TO FEATURE STORE
    # -----------------------------
    fs = get_feature_store()

    create_feature_group(
        fs,
        name="drivers_details_fg",
        version=1,
        df=drivers_df,
        primary_key=["driver_id"],
        event_time="event_time",
        online_enabled=False,
    )

    create_feature_group(
        fs,
        name="truck_details_fg",
        version=1,
        df=trucks_df,
        primary_key=["truck_id"],
        event_time="event_time",
        online_enabled=False,
    )

    create_feature_group(
        fs,
        name="routes_details_fg",
        version=1,
        df=routes_df,
        primary_key=["route_id"],
        event_time="event_time",
        online_enabled=False,
    )

    create_feature_group(
        fs,
        name="city_weather_details_fg",
        version=1,
        df=weather_df,
        primary_key=["city_id", "hour"],
        event_time="date",
        online_enabled=True,
    )

    logger.info("✅ Pipeline completed successfully")


# -------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()

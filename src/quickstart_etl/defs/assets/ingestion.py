# import base64
# import json
# import os
# from io import BytesIO

# import matplotlib.pyplot as plt
# import pandas as pd
# import requests

from dagster import (
    MaterializeResult,
    asset,
)

from quickstart_etl.partitions import daily_partitions

# kinds=["python", "gcs", "api", "bigquery", "io-manager"],


@asset(group_name="ingestion", partitions_def=daily_partitions)
def raw_airline_market_data(context) -> MaterializeResult:
    """Pull daily OHLCV for airline tickers from Yahoo Finance."""
    # DAL (Delta), UAL (United), AAL (American), LUV (Southwest)
    # These are Fetcherr's actual customers / industry
    ...


@asset(group_name="ingestion", partitions_def=daily_partitions)
def raw_weather_data(context) -> MaterializeResult:
    """Pull daily weather for major airline hub cities from Open-Meteo."""
    # ATL, LAX, ORD, DFW, JFK — temperature, precipitation, wind
    # External signal, just like Fetcherr ingests weather
    ...


@asset(group_name="ingestion", partitions_def=daily_partitions)
def raw_currency_rates(context) -> MaterializeResult:
    """Pull daily USD exchange rates from Frankfurter API."""
    # EUR, GBP, BRL, MAD — currencies of Fetcherr's airline partners
    # (Virgin Atlantic = GBP, Azul = BRL, Royal Air Maroc = MAD)
    ...

import logging
import os
from datetime import datetime
from typing import Tuple

import requests

from app.utils.config import ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY, ALPACA_BASE_URL

logger = logging.getLogger(__name__)


def _get_headers() -> dict:
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY_ID,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET_KEY,
    }


def check_market_open() -> Tuple[bool, str]:
    """
    Check if the market is open today using Alpaca Clock API.

    Returns
    -------
    Tuple[bool, str]
        (is_open, message)
    """
    try:
        # A better endpoint is /v2/calendar
        return _check_calendar_open()

    except Exception as e:
        logger.error(f"Error checking market clock: {e}")
        return False, str(e)


def _check_calendar_open() -> Tuple[bool, str]:
    """Check calendar for today."""
    url = f"{ALPACA_BASE_URL}/v2/calendar"
    today_str = datetime.utcnow().strftime("%Y-%m-%d")

    try:
        params = {"start": today_str, "end": today_str}
        response = requests.get(url, headers=_get_headers(), params=params, timeout=10)

        if response.status_code != 200:
            return False, f"Calendar API Error: {response.status_code}"

        data = response.json()
        if not data:
            return False, "Market Closed (No calendar entry for today)"

        # If we get a result, the market is open today
        session = data[0]
        open_time = session.get("open")
        close_time = session.get("close")

        return True, f"Market Open Today ({open_time} - {close_time})"

    except Exception as e:
        logger.error(f"Error checking calendar: {e}")
        return False, str(e)


def validate_model_age(model_path: str, max_days: int = 6) -> Tuple[bool, str]:
    """
    Check if the model file is fresh enough.

    Parameters
    ----------
    model_path : str
        Path to the model file.
    max_days : int
        Maximum allowed age in days.

    Returns
    -------
    Tuple[bool, str]
        (is_valid, message)
    """
    if not os.path.exists(model_path):
        return False, "Model file not found"

    try:
        mtime = os.path.getmtime(model_path)
        model_date = datetime.fromtimestamp(mtime)
        age = datetime.now() - model_date

        if age.days > max_days:
            return (
                False,
                f"Model is too old ({age.days} days). Max allowed: {max_days}.",
            )

        return True, f"Model is fresh ({age.days} days old)."

    except Exception as e:
        return False, f"Error checking model age: {e}"

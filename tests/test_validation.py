"""
Unit tests for validation service (T015).
"""

import os
import time
from unittest.mock import MagicMock, patch

from app.services.validation import check_market_open, validate_model_age


@patch("app.services.validation.requests.get")
def test_check_market_open_api_success(mock_get):
    """Test market open check with successful API response."""
    # Mock calendar response (clock endpoint is no longer called)
    mock_cal_resp = MagicMock()
    mock_cal_resp.status_code = 200
    mock_cal_resp.json.return_value = [{"open": "09:30", "close": "16:00"}]

    mock_get.return_value = mock_cal_resp

    is_open, msg = check_market_open()
    assert is_open is True
    assert "Market Open Today" in msg


@patch("app.services.validation.requests.get")
def test_check_market_open_api_fail(mock_get):
    """Test market open check with API failure."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_get.return_value = mock_resp

    is_open, msg = check_market_open()
    assert is_open is False
    assert "API Error" in msg


def test_validate_model_age_success(tmp_path):
    """Test model age validation with fresh model."""
    model_path = tmp_path / "model.pkl"
    model_path.touch()

    # Ensure mtime is now
    os.utime(model_path, None)

    is_valid, msg = validate_model_age(str(model_path), max_days=6)
    assert is_valid is True
    assert "fresh" in msg


def test_validate_model_age_old(tmp_path):
    """Test model age validation with old model."""
    model_path = tmp_path / "model.pkl"
    model_path.touch()

    # Set mtime to 10 days ago
    past_time = time.time() - (10 * 24 * 3600)
    os.utime(model_path, (past_time, past_time))

    is_valid, msg = validate_model_age(str(model_path), max_days=6)
    assert is_valid is False
    assert "too old" in msg


def test_validate_model_age_missing():
    """Test model age validation with missing file."""
    is_valid, msg = validate_model_age("non_existent.pkl")
    assert is_valid is False
    assert "not found" in msg

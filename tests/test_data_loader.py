"""
Unit tests for data loader service (T012).
"""

from unittest.mock import MagicMock, patch

import pandas as pd

from app.services.data_loader import download_latest_snapshot, load_and_merge_data


@patch("app.services.data_loader.subprocess.run")
def test_download_latest_snapshot_success(mock_run, tmp_path):
    """Test successful snapshot download."""
    # Mock gh version check
    mock_run.return_value = MagicMock(returncode=0)

    # We need to simulate file movement, but since we mock subprocess,
    # the files won't be created by 'gh'.
    # We can mock Path.rename or just ensure subprocess was called correctly.
    # However, the function checks if files exist before renaming.
    # So we should probably mock Path.exists and Path.rename too?
    # Or just mock the whole file system operations.

    with patch("app.services.data_loader.Path") as mock_path:
        # Setup existence checks
        mock_path_obj = MagicMock()
        mock_path.return_value = mock_path_obj

        # When checking if temp files exist
        mock_path_obj.__truediv__.return_value.exists.return_value = True

        result = download_latest_snapshot(output_dir="data", model_dir="models")

        assert result is True
        assert mock_run.call_count >= 2  # version check + download


@patch("app.services.data_loader.subprocess.run")
def test_download_latest_snapshot_no_gh(mock_run):
    """Test download when gh CLI is missing."""
    mock_run.side_effect = FileNotFoundError

    result = download_latest_snapshot()
    assert result is False


@patch("app.services.data_loader.os.path.exists")
@patch("app.services.data_loader.pd.read_parquet")
@patch("app.services.data_loader.fetch_stock_data_alpaca")
def test_load_and_merge_data_up_to_date(mock_fetch, mock_read_parquet, mock_exists):
    """Test loading data when snapshot is up to date."""
    mock_exists.return_value = True

    # Snapshot up to today (so no incremental fetch needed)
    # Use pd.Timestamp to match what's in the parquet file
    today = pd.Timestamp.utcnow().normalize()

    mock_df = pd.DataFrame({"timestamp": [today], "symbol": ["AAPL"], "close": [150.0]})
    mock_read_parquet.return_value = mock_df

    # Should not fetch new data (start_date > today check)
    result = load_and_merge_data(symbols=["AAPL"])

    assert len(result) == 1
    mock_fetch.assert_not_called()


@patch("app.services.data_loader.os.path.exists")
@patch("app.services.data_loader.pd.read_parquet")
@patch("app.services.data_loader.fetch_stock_data_alpaca")
def test_load_and_merge_data_incremental(mock_fetch, mock_read_parquet, mock_exists):
    """Test loading data with incremental fetch."""
    mock_exists.return_value = True

    # Snapshot from 3 days ago to ensure incremental fetch
    today = pd.Timestamp.utcnow().normalize()
    three_days_ago = today - pd.Timedelta(days=3)

    mock_history = pd.DataFrame(
        {"timestamp": [three_days_ago], "symbol": ["AAPL"], "close": [150.0]}
    )
    mock_read_parquet.return_value = mock_history

    # Fetch returns new bar
    mock_new_bar = pd.DataFrame(
        {"timestamp": [today], "symbol": ["AAPL"], "close": [152.0]}
    )
    mock_fetch.return_value = mock_new_bar

    result = load_and_merge_data(symbols=["AAPL"])

    # Should have merged data
    assert len(result) == 2
    # Should have called fetch for AAPL
    assert mock_fetch.call_count == 1
    assert result.iloc[-1]["close"] == 152.0

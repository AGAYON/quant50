import sys
from unittest.mock import MagicMock, patch

from jobs import daily_run, weekly_train


@patch("jobs.daily_run.generate_error_report")
@patch("jobs.daily_run.cancel_all_orders")
@patch("jobs.daily_run.pipeline.run_pipeline")
@patch("jobs.daily_run.load_and_merge_data")
@patch("jobs.daily_run.validate_model_age")
@patch("jobs.daily_run.download_latest_snapshot")
@patch("jobs.daily_run.check_market_open")
def test_daily_run_success(
    mock_check_open,
    mock_download,
    mock_validate_age,
    mock_load_data,
    mock_run_pipeline,
    mock_cancel,
    mock_report,
):
    # Setup mocks for success path
    mock_check_open.return_value = (True, "Open")
    mock_download.return_value = True
    mock_validate_age.return_value = (True, "Fresh")
    mock_load_data.return_value = MagicMock(empty=False)
    mock_run_pipeline.return_value = {"status": "success"}

    with patch.object(sys, 'argv', ["daily_run.py"]):
        exit_code = daily_run.main()

    assert exit_code == 0
    mock_run_pipeline.assert_called_once()
    mock_cancel.assert_not_called()


@patch("jobs.daily_run.generate_error_report")
@patch("jobs.daily_run.cancel_all_orders")
@patch("jobs.daily_run.pipeline.run_pipeline")
@patch("jobs.daily_run.load_and_merge_data")
@patch("jobs.daily_run.validate_model_age")
@patch("jobs.daily_run.download_latest_snapshot")
@patch("jobs.daily_run.check_market_open")
def test_daily_run_failure(
    mock_check_open,
    mock_download,
    mock_validate_age,
    mock_load_data,
    mock_run_pipeline,
    mock_cancel,
    mock_report,
):
    # Setup mocks for failure path (pipeline error)
    mock_check_open.return_value = (True, "Open")
    mock_download.return_value = True
    mock_validate_age.return_value = (True, "Fresh")
    mock_load_data.return_value = MagicMock(empty=False)
    mock_run_pipeline.return_value = {"error": "Pipeline failed"}

    with patch.object(sys, 'argv', ["daily_run.py"]):
        exit_code = daily_run.main()

    assert exit_code == 1
    mock_cancel.assert_called_once()  # Safety Halt triggered
    mock_report.assert_called_once()  # Error report generated


@patch("jobs.weekly_train._load_bars_from_duckdb")
@patch("jobs.weekly_train.generate_features")
@patch("jobs.weekly_train.build_labels_excess_return")
@patch("jobs.weekly_train.build_training_dataset")
@patch("jobs.weekly_train.fit")
@patch("jobs.weekly_train.save_model_artifacts")
def test_weekly_train_success(
    mock_save,
    mock_fit,
    mock_build_ds,
    mock_build_labels,
    mock_gen_features,
    mock_load_bars,
):
    # Setup mocks
    mock_load_bars.return_value = MagicMock(empty=False)
    mock_gen_features.return_value = MagicMock(reset_index=lambda: MagicMock())
    mock_fit.return_value = {"metrics": {}}

    with patch.object(sys, 'argv', ["weekly_train.py", "--dry-run"]):
        # We catch SystemExit because main() calls sys.exit() via
        # raise SystemExit(main())
        # in if __name__ == "__main__"
        # But here we call main() directly which returns int
        exit_code = weekly_train.main()

    assert exit_code == 0
    mock_fit.assert_called_once()
    # save should not be called in dry-run
    mock_save.assert_not_called()


@patch("jobs.weekly_train._load_bars_from_duckdb")
def test_weekly_train_no_data(mock_load_bars):
    mock_load_bars.return_value = MagicMock(empty=True)

    with patch.object(sys, 'argv', ["weekly_train.py"]):
        exit_code = weekly_train.main()

    assert exit_code == 0

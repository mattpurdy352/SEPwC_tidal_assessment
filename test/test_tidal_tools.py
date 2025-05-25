import pytest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0,"../")
sys.path.insert(0,"./")
from datetime import datetime, timedelta, timezone # Added timezone for epoch
from unittest.mock import patch, MagicMock, ANY

# --- Fixtures ---
@pytest.fixture
def sample_df_empty():
    return pd.DataFrame({'Sea Level': pd.Series(dtype=float)},
                        index=pd.to_datetime([]).tz_localize('UTC')) # Timezone-aware

@pytest.fixture
def sample_df_insufficient():
    start_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
    time_index = pd.to_datetime([start_time + timedelta(hours=i) for i in range(3)], )
    # Only 1 valid point, if constituents need e.g. 2*2=4 points
    sea_level = np.array([1.0, np.nan, np.nan])
    return pd.DataFrame({'Sea Level': sea_level}, index=time_index)

@pytest.fixture
def sample_df_valid():
    start_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
    # Enough data for 2 constituents (e.g., 2*2=4 points needed, provide more)
    time_index = pd.to_datetime([start_time + timedelta(hours=i) for i in range(10)])
    sea_level = np.arange(10, dtype=float)
    return pd.DataFrame({'Sea Level': sea_level}, index=time_index)

@pytest.fixture
def sample_epoch():
    return datetime(2023, 1, 1, tzinfo=timezone.utc) # Timezone-aware

@pytest.fixture
def sample_constituents():
    return ['M2', 'S2']

@pytest.fixture
def sample_latitude():
    return 57.0

# --- Tests for reconstruct_tide ---

# Assuming `reconstruct_tide` itself imports or defines `solve` and `reconstruct`
# If they are global in the module `tidal_tools`, patching 'tidal_tools.solve' is correct.
@patch('tidal_tools.solve')
@patch('tidal_tools.reconstruct')
def test_reconstruct_tide_empty_data(mock_utide_reconstruct, mock_utide_solve,
                                     sample_df_empty, sample_constituents, sample_epoch, sample_latitude):
    from tidal_tools import reconstruct_tide # Import here or ensure discoverable
    reconstructed, coef = reconstruct_tide(
        sample_df_empty, sample_constituents, sample_epoch, sample_latitude
    )
    assert isinstance(reconstructed, np.ndarray) and reconstructed.size == 0
    assert isinstance(coef, dict) and not coef
    mock_utide_solve.assert_not_called()
    mock_utide_reconstruct.assert_not_called()

@patch('tidal_tools.solve')
@patch('tidal_tools.reconstruct')
def test_reconstruct_tide_insufficient_data(mock_utide_reconstruct, mock_utide_solve,
                                            sample_df_insufficient, sample_constituents,
                                            sample_epoch, sample_latitude, capsys):
    from tidal_tools import reconstruct_tide
    # sample_df_insufficient has 1 valid point. len(constituents)*2 = 4.
    expected_time_hours_len = len(sample_df_insufficient)
    reconstructed, coef = reconstruct_tide(
        sample_df_insufficient, sample_constituents, sample_epoch, sample_latitude
    )
    
    captured = capsys.readouterr()
    assert "Warning: not enough valid data points" in captured.out
    assert np.all(np.isnan(reconstructed))
    assert reconstructed.shape == (expected_time_hours_len,)
    assert isinstance(coef, dict) and not coef
    mock_utide_solve.assert_not_called()
    mock_utide_reconstruct.assert_not_called()

@patch('tidal_tools.solve')
@patch('tidal_tools.reconstruct')
def test_reconstruct_tide_success(mock_utide_reconstruct, mock_utide_solve,
                                  sample_df_valid, sample_constituents,
                                  sample_epoch, sample_latitude):
    from tidal_tools import reconstruct_tide
    # Mock return values
    mock_coef_obj = MagicMock(name="MockCoef")
    mock_utide_solve.return_value = mock_coef_obj

    mock_tide_h_data = np.random.rand(len(sample_df_valid))
    mock_tide_obj = MagicMock(name="MockTide")
    mock_tide_obj.h = mock_tide_h_data
    mock_utide_reconstruct.return_value = mock_tide_obj

    reconstructed, coef_out = reconstruct_tide(
        sample_df_valid, sample_constituents, sample_epoch, sample_latitude
    )

    # Check that solve was called (assuming valid data passes the length check)
    assert mock_utide_solve.call_count == 1
    call_args, call_kwargs = mock_utide_solve.call_args
    # More detailed checks for args of solve:
    # Expected time_hours_valid and y_valid
    time_hours_expected = (sample_df_valid.index - sample_epoch).total_seconds() / 3600.0
    y_expected = sample_df_valid['Sea Level'].values
    valid_indices_expected = ~np.isnan(y_expected)

    assert np.array_equal(call_args[0], time_hours_expected[valid_indices_expected])
    assert np.array_equal(call_args[1], y_expected[valid_indices_expected])
    assert call_kwargs['constit'] == sample_constituents
    assert call_kwargs['method'] == 'ols'
    assert not call_kwargs['nodal']
    assert not call_kwargs['trend']
    
    mock_utide_reconstruct.assert_called_once_with(ANY, mock_coef_obj) # ANY for full time_hours
    assert np.array_equal(reconstructed, mock_tide_h_data)
    assert coef_out is mock_coef_obj


# --- Tests for plot_tide_fit ---
@patch('tidal_tools.plt') # Assuming plot_tide_fit uses plt imported in tidal_tools
def test_plot_tide_fit_display(mock_plt, sample_df_valid):
    from tidal_tools import plot_tide_fit
    reconstructed_data = np.random.rand(len(sample_df_valid))
    title = "Display Test"
    
    plot_tide_fit(sample_df_valid, reconstructed_data, title=title, save_to_file=False)

    mock_plt.figure.assert_called_once_with(figsize=(15, 6))
    # Check for at least 3 plot calls: observed, reconstructed, residuals
    assert mock_plt.plot.call_count >= 3
    mock_plt.xlabel.assert_called_with("Time")
    mock_plt.ylabel.assert_called_with("Sea Level (m)")
    mock_plt.title.assert_called_with(title)
    mock_plt.legend.assert_called_once()
    mock_plt.grid.assert_called_with(True)
    mock_plt.tight_layout.assert_called_once()
    mock_plt.show.assert_called_once()
    mock_plt.savefig.assert_not_called()
    mock_plt.close.assert_not_called()

@patch('tidal_tools.plt')
def test_plot_tide_fit_save_file(mock_plt, sample_df_valid):
    from tidal_tools import plot_tide_fit
    reconstructed_data = np.random.rand(len(sample_df_valid))
    title = "Save Test / File"
    expected_filename = "Save_Test___File.png"

    plot_tide_fit(sample_df_valid, reconstructed_data, title=title, save_to_file=True)

    mock_plt.savefig.assert_called_once_with(expected_filename)
    mock_plt.close.assert_called_once()
    mock_plt.show.assert_not_called()

@patch('tidal_tools.plt')
def test_plot_tide_fit_residuals_mismatch(mock_plt, sample_df_valid, capsys):
    from tidal_tools import plot_tide_fit
    reconstructed_data = np.random.rand(len(sample_df_valid) - 1) # Mismatched length
    
    plot_tide_fit(sample_df_valid, reconstructed_data)
    
    # Count calls for observed and reconstructed (should be 2)
    actual_plot_calls = mock_plt.plot.call_args_list
    plot_labels = [call.kwargs.get('label') for call in actual_plot_calls if isinstance(call.kwargs, dict)]

    assert 'Observed' in plot_labels
    assert 'Reconstructed' in plot_labels
    assert 'Residual' not in plot_labels # Residual plot should be skipped

    captured = capsys.readouterr()
    assert "Warning: Reconstructed data length mismatch. Residuals not plotted." in captured.out


# --- Tests for _perform_reconstruction_and_plot ---
# These require patching functions called by _perform_reconstruction_and_plot
# and also the global 'solve' and 'reconstruct' that it checks.

@patch('tidal_tools.plot_tide_fit')
@patch('tidal_tools.reconstruct_tide')
def test_perform_no_plot_no_save(mock_reconstruct_tide, mock_plot_tide_fit,
                                 sample_df_valid, sample_epoch, sample_latitude, capsys):
    from tidal_tools import _perform_reconstruction_and_plot
    # Assuming solve and reconstruct are available (not None) for this test path
    # if they were imported globally in tidal_tools
    with patch('tidal_tools.solve', MagicMock()), patch('tidal_tools.reconstruct', MagicMock()):
        _perform_reconstruction_and_plot(
            sample_df_valid, sample_epoch, sample_latitude,
            is_verbose=True, do_plot=False, do_save=False
        )
    mock_reconstruct_tide.assert_not_called()
    mock_plot_tide_fit.assert_not_called()
    captured = capsys.readouterr()
    assert "Plotting and saving not requested" in captured.out

@patch('tidal_tools.plot_tide_fit')
@patch('tidal_tools.reconstruct_tide')
def test_perform_empty_data(mock_reconstruct_tide, mock_plot_tide_fit,
                            sample_df_empty, sample_epoch, sample_latitude, capsys):
    from tidal_tools import _perform_reconstruction_and_plot
    with patch('tidal_tools.solve', MagicMock()), patch('tidal_tools.reconstruct', MagicMock()):
        _perform_reconstruction_and_plot(
            sample_df_empty, sample_epoch, sample_latitude,
            is_verbose=True, do_plot=True, do_save=False
        )
    mock_reconstruct_tide.assert_not_called()
    mock_plot_tide_fit.assert_not_called()
    captured = capsys.readouterr()
    # The function prints to sys.stderr
    assert "Warning: Data section is empty or all NaN" in captured.err

@patch('tidal_tools.plot_tide_fit')
@patch('tidal_tools.reconstruct_tide')
def test_perform_utide_not_available(mock_reconstruct_tide, mock_plot_tide_fit,
                                     sample_df_valid, sample_epoch, sample_latitude, capsys):
    from tidal_tools import _perform_reconstruction_and_plot
    # Simulate utide not being available by patching solve and reconstruct to None
    with patch('tidal_tools.solve', None), patch('tidal_tools.reconstruct', None):
        _perform_reconstruction_and_plot(
            sample_df_valid, sample_epoch, sample_latitude,
            is_verbose=True, do_plot=True, do_save=False
        )
    mock_reconstruct_tide.assert_not_called()
    mock_plot_tide_fit.assert_not_called()
    captured = capsys.readouterr()
    assert "utide library not available. Skipping tidal reconstruction." in captured.err

@patch('tidal_tools.plot_tide_fit')
@patch('tidal_tools.reconstruct_tide')
def test_perform_latitude_missing_for_utide(mock_reconstruct_tide, mock_plot_tide_fit,
                                             sample_df_valid, sample_epoch, capsys):
    from tidal_tools import _perform_reconstruction_and_plot
    # Ensure solve and reconstruct appear available
    with patch('tidal_tools.solve', MagicMock()), patch('tidal_tools.reconstruct', MagicMock()):
        _perform_reconstruction_and_plot(
            sample_df_valid, sample_epoch, latitude=None, # Latitude is missing
            is_verbose=True, do_plot=True, do_save=False
        )
    mock_reconstruct_tide.assert_not_called() # Should not be called due to missing latitude
    mock_plot_tide_fit.assert_not_called()
    captured = capsys.readouterr()
    assert "Error: Latitude is required for utide tidal reconstruction" in captured.err

@patch('tidal_tools.plot_tide_fit')
@patch('tidal_tools.reconstruct_tide')
def test_perform_reconstruction_success_plot(mock_reconstruct_tide, mock_plot_tide_fit,
                                             sample_df_valid, sample_epoch, sample_latitude):
    from tidal_tools import _perform_reconstruction_and_plot
    mock_signal = np.array([1.0] * len(sample_df_valid))
    mock_coeffs = {"M2": "data"}
    mock_reconstruct_tide.return_value = (mock_signal, mock_coeffs)

    with patch('tidal_tools.solve', MagicMock()), patch('tidal_tools.reconstruct', MagicMock()):
        _perform_reconstruction_and_plot(
            sample_df_valid, sample_epoch, sample_latitude,
            is_verbose=False, do_plot=True, do_save=False
        )
    
    mock_reconstruct_tide.assert_called_once_with(
        sample_df_valid, ['M2', 'S2'], sample_epoch, sample_latitude
    )
    mock_plot_tide_fit.assert_called_once_with(
        sample_df_valid, mock_signal, title=ANY, save_to_file=False
    )

@patch('tidal_tools.plot_tide_fit')
@patch('tidal_tools.reconstruct_tide')
def test_perform_reconstruction_failure(mock_reconstruct_tide, mock_plot_tide_fit,
                                        sample_df_valid, sample_epoch, sample_latitude, capsys):
    from tidal_tools import _perform_reconstruction_and_plot
    mock_reconstruct_tide.side_effect = Exception("Test Reconstruction Error")

    with patch('tidal_tools.solve', MagicMock()), patch('tidal_tools.reconstruct', MagicMock()):
        _perform_reconstruction_and_plot(
            sample_df_valid, sample_epoch, sample_latitude,
            is_verbose=False, do_plot=True, do_save=False
        )
    
    mock_reconstruct_tide.assert_called_once()
    mock_plot_tide_fit.assert_not_called()
    captured = capsys.readouterr()
    assert "Error during tidal reconstruction: Test Reconstruction Error" in captured.err

@patch('tidal_tools.plot_tide_fit')
@patch('tidal_tools.reconstruct_tide')
def test_perform_reconstruction_returns_none(mock_reconstruct_tide, mock_plot_tide_fit,
                                             sample_df_valid, sample_epoch, sample_latitude, capsys):
    from tidal_tools import _perform_reconstruction_and_plot
    mock_reconstruct_tide.return_value = (None, {}) # Signal is None

    with patch('tidal_tools.solve', MagicMock()), patch('tidal_tools.reconstruct', MagicMock()):
        _perform_reconstruction_and_plot(
            sample_df_valid, sample_epoch, sample_latitude,
            is_verbose=True, do_plot=True, do_save=False
        )
    mock_reconstruct_tide.assert_called_once()
    mock_plot_tide_fit.assert_not_called()
    captured = capsys.readouterr()
    assert "Warning: Tidal reconstruction did not produce a signal." in captured.err

@patch('tidal_tools.plot_tide_fit')
@patch('tidal_tools.reconstruct_tide')
def test_perform_reconstruction_all_nan_signal(mock_reconstruct_tide, mock_plot_tide_fit,
                                               sample_df_valid, sample_epoch, sample_latitude, capsys):
    from tidal_tools import _perform_reconstruction_and_plot
    mock_signal_all_nan = np.full(len(sample_df_valid), np.nan)
    mock_coeffs = {"M2": "data"}
    mock_reconstruct_tide.return_value = (mock_signal_all_nan, mock_coeffs)

    with patch('tidal_tools.solve', MagicMock()), patch('tidal_tools.reconstruct', MagicMock()):
        _perform_reconstruction_and_plot(
            sample_df_valid, sample_epoch, sample_latitude,
            is_verbose=False, do_plot=True, do_save=False # Plotting is requested
        )
    
    mock_reconstruct_tide.assert_called_once()
    # plot_tide_fit will still be called, but a warning about all NaN should be printed by _perform...
    mock_plot_tide_fit.assert_called_once()
    captured = capsys.readouterr()
    assert "Warning: Tidal reconstruction resulted in all NaN values." in captured.err

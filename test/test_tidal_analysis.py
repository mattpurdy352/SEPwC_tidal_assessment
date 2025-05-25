import pytest
import pandas as pd
import pytz
from datetime import datetime as dt  
import tidal_analysis as tidal_analysis_module


class TestTidalModule:
    def test_utide_is_available(self, monkeypatch):
        """
        Tests that _check_utide_availability does NOT raise an error
        if utide.solve and utide.reconstruct are mocked as available at the module level.
        """
        def mock_solve_function(*args, **kwargs):
            pass # Dummy function
        
        def mock_reconstruct_function(*args, **kwargs):
            pass # Dummy function

        # Patch the 'solve' and 'reconstruct' attributes of the imported module
        monkeypatch.setattr(tidal_analysis_module, 'solve', mock_solve_function)
        monkeypatch.setattr(tidal_analysis_module, 'reconstruct', mock_reconstruct_function)

        try:
            tidal_analysis_module._check_utide_availability() # Call using the module alias
        except EnvironmentError:
            pytest.fail("EnvironmentError was raised even though utide components were mocked as available.")


    def test_prepare_inputs_same_timezone(self):
        """
        Tests basic functionality when data.index and start_datetime_epoch 
        are already in the same timezone (e.g. UTC).
        """
        utc_tz = pytz.utc
        data_timestamps_utc = pd.to_datetime([
            '2023-01-01 00:00:00',
            '2023-01-01 01:00:00',
        ]).tz_localize(utc_tz)
        
        sample_data_df = pd.DataFrame(
            {'Sea Level': [1.0, 1.5]},
            index=data_timestamps_utc
        )
        start_epoch_utc = utc_tz.localize(dt(2023, 1, 1, 0, 0, 0))

        time_array, _ = tidal_analysis_module._prepare_tidal_analysis_inputs(
            sample_data_df,
            start_epoch_utc
        )
        
        assert time_array[0] == pytest.approx(0.0)
        assert time_array[1] == pytest.approx(1.0 / 24.0)
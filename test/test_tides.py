import pytest
import sys
sys.path.insert(0,"../")
sys.path.insert(0,"./")
from tidal_analysis import *
from pylint.lint import Run
from pylint.reporters import CollectingReporter
from dataclasses import asdict
import pandas as pd
import numpy as np
import datetime 
import pytz

class TestTidalAnalysis():
    
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
    
    def test_reading_data(self):
        tidal_file = "data/1947ABE.txt"
        
        data = read_tidal_data(tidal_file)
        assert "Sea Level" in data.columns
        assert type(data.index) == pd.core.indexes.datetimes.DatetimeIndex
        assert data['Sea Level'].size == 8760
        assert '1947-01-01 00:00:00' in data.index
        assert '1947-12-31 23:00:00' in data.index

        # check for M, N and T data; should be NaN
        assert data['Sea Level'].isnull().any()
        assert pd.api.types.is_float_dtype(data['Sea Level'])
        
        # check for error on unknown file
        with pytest.raises(FileNotFoundError):
            read_tidal_data("missing_file.dat")

    
    def test_join_data(self):

        df_no_sea_level_1 = pd.DataFrame({'colA': [1, 2]}, index=pd.to_datetime(['20230101', '20230102']))
        df_no_sea_level_2 = pd.DataFrame({'colB': [3, 4]}, index=pd.to_datetime(['20230103', '20230104']))
       
         with pytest.raises(ValueError, match="Both datasets for join_data must contain a 'Sea Level' column."):
          join_data(df_no_sea_level_1, df_no_sea_level_2)
        gauge_files = ['data/1946ABE.txt', 'data/1947ABE.txt']
        data1_valid = read_tidal_data(gauge_files[1]) # 1947 data
        data2_valid = read_tidal_data(gauge_files[0]) # 1946 data
        
        data = join_data(data1_valid, data2_valid)

        assert "Sea Level" in data.columns
        assert isinstance(data.index, pd.core.indexes.datetimes.DatetimeIndex)
        assert data['Sea Level'].size == 8760*2

        # check sorting (we join 1947 to 1946, but expect 1946 to 1947)
        assert data.index[0] == pd.Timestamp('1946-01-01 00:00:00', tz='UTC')
        assert data.index[-1] == pd.Timestamp('1947-12-31 23:00:00', tz='UTC')

        data2_modified_for_test = read_tidal_data(gauge_files[0]) 
        data2_modified_for_test_no_sea_level = data2_modified_for_test.drop(columns=["Sea Level"])

        with pytest.raises(ValueError, match="Both datasets for join_data must contain a 'Sea Level' column."):
         join_data(data1_valid, data2_modified_for_test_no_sea_level)
        
    
    def test_extract_year(self):
        
        gauge_files = ['data/1946ABE.txt', 'data/1947ABE.txt']

        data1 = read_tidal_data(gauge_files[1])
        data2 = read_tidal_data(gauge_files[0])
        data = join_data(data1, data2)
        
        year1947 = extract_single_year_remove_mean("1947",data)
        assert "Sea Level" in year1947.columns
        assert type(year1947.index) == pd.core.indexes.datetimes.DatetimeIndex
        assert year1947['Sea Level'].size == 8760

        mean = np.mean(year1947['Sea Level'])
        # check mean is near zero
        assert mean == pytest.approx(0)

        # check something sensible when a year is given that doesn't exist

    def test_extract_section(self):

        gauge_files = ['data/1946ABE.txt', 'data/1947ABE.txt']

        data1 = read_tidal_data(gauge_files[1])
        data2 = read_tidal_data(gauge_files[0])
        data = join_data(data1, data2)
        
        year1946_47 = extract_section_remove_mean("19461215", "19470310", data)
        assert "Sea Level" in year1946_47.columns
        assert isinstance(year1946_47.index, pd.DatetimeIndex)
        assert year1946_47['Sea Level'].size == 2064

        mean = np.mean(year1946_47['Sea Level'])
        # check mean is near zero
        assert mean == pytest.approx(0)

        data_segment = extract_section_remove_mean("19470115", "19470310", data1)
        assert "Sea Level" in data_segment.columns
        assert isinstance(data_segment.index, pd.DatetimeIndex)
        assert data_segment['Sea Level'].size == 1320

        mean = np.mean(data_segment['Sea Level'])
        # check mean is near zero
        assert mean == pytest.approx(0)

        # check something sensible is done when dates are formatted correctly.

    def test_correct_tides(self):

        gauge_files = ['data/1946ABE.txt', 'data/1947ABE.txt']
        data1 = read_tidal_data(gauge_files[1])
        data2 = read_tidal_data(gauge_files[0])
        data = join_data(data1, data2)

        data_segment =extract_section_remove_mean("19460115", "19470310", data)

        constituents  = ['M2', 'S2']
        tz = pytz.timezone("utc")
        start_datetime = datetime.datetime(1946,1,15,0,0,0, tzinfo=tz)
        amp,pha = tidal_analysis(data_segment, constituents, start_datetime)

        # for Aberdeen, the M2 and S2 amps are 1.307 and 0.441
        assert amp[0] == pytest.approx(1.307,abs=0.1)
        assert amp[1] == pytest.approx(0.441,abs=0.1)


    def test_linear_regression(self):

        gauge_files = ['data/1946ABE.txt', 'data/1947ABE.txt']
        data1 = read_tidal_data(gauge_files[1])
        data2 = read_tidal_data(gauge_files[0])
        data = join_data(data1, data2)

        slope, p_value = sea_level_rise(data)
        
        assert slope == pytest.approx(2.94e-05, abs=1e-6)
        assert p_value == pytest.approx(0.427,abs=0.1)
        

    def test_lint(self):
        files =  ["tidal_analysis.py"]
        #pylint_options = ["--disable=line-too-long,import-error,fixme"]
        pylint_options = []

        report = CollectingReporter()
        result = Run(
                    files,
                    reporter=report,
                    exit=False,
                )
        score = result.linter.stats.global_note
        nErrors = len(report.messages)

        print("Score: " + str(score))
        line_format = "{path}:{line}:{column}: {msg_id}: {msg} ({symbol})"
        for error in report.messages:
            print(line_format.format(**asdict(error)))   

        assert nErrors < 500
        assert nErrors < 400
        assert nErrors < 250
        assert nErrors < 100
        assert nErrors < 50
        assert nErrors < 10
        assert nErrors == 0
        assert score > 3
        assert score > 5
        assert score > 7
        assert score > 9

class TestRegression():

    def test_whitby_regression(self):

        from subprocess import run
        result = run(["python","tidal_analysis.py","-v","data/whitby"], capture_output=True, check=True)
        assert len(result.stdout) > 25

    def test_aberdeen_regression(self):

        from subprocess import run
        result = run(["python","tidal_analysis.py","--verbose","data/aberdeen"], capture_output=True, check=True)
        assert len(result.stdout) > 25

    def test_dover_regression(self):

        from subprocess import run
        result = run(["python","tidal_analysis.py","-v","data/dover"], capture_output=True, check=True)
        assert len(result.stdout) > 25


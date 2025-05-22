#!/usr/bin/env python3
import argparse
import os
from datetime import datetime # datetime class from datetime module
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from scipy.stats import linregress

try:
    from utide import solve, reconstruct
except ImportError:
    print("Warning: 'utide' library not found. "
          "Functions 'solve' and 'reconstruct' will not be available.")
    solve = None
    reconstruct = None

SECONDS_PER_HOUR = 3600.0
MIN_DATAPOINTS_PER_CONSTITUENT = 2 
NUM_FILE_HEADER_LINES_TO_SKIP: int = 11
# Expected column names after skipping headers and providing names to read_csv
EXPECTED_RAW_COLUMN_NAMES: list[str] = [
    'Cycle', 'Date_str', 'Time_str', 'Sea_Level_Raw', 'Residual_Raw'
]
# Datetime format string used for parsing combined date and time
DATETIME_FORMAT_STR: str = '%Y/%m/%d %H:%M:%S'

def _check_utide_availability() -> None:
    """
    Checks if the UTide 'solve' and 'reconstruct' functions are available.

    Raises:
        EnvironmentError: If 'solve' or 'reconstruct' from UTide is not loaded.
    """
    if solve is None or reconstruct is None:
        error_message = (
            "'utide' library functions ('solve' and/or 'reconstruct') are "
            "not available. Please ensure utide is installed and imported "
            "correctly."
        )
        raise EnvironmentError(error_message)

def _prepare_tidal_analysis_inputs(
    data: pd.DataFrame,
    start_datetime_epoch: datetime
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepares time (in hours from epoch) and sea level arrays for tidal analysis.

    Handles timezone consistency between the data's DatetimeIndex and the
    start_datetime_epoch. Assumes data.index is already localized (e.g., to UTC).

    Args:
        data (pd.DataFrame): Input DataFrame with a DatetimeIndex and
                             'Sea Level' column.
        start_datetime_epoch (datetime): The reference datetime (t=0) for the
                                         analysis. Must be timezone-aware.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            time_hours (np.ndarray): Time in hours since start_datetime_epoch.
            sea_level_values (np.ndarray): NumPy array of sea level values.

    Raises:
        TypeError: If data.index is not a DatetimeIndex.
        ValueError: If timezone consistency cannot be achieved or data is unsuitable.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Input 'data' must have a pandas DatetimeIndex.")

    pd_start_epoch = pd.Timestamp(start_datetime_epoch)

    if data.index.tz is None:
        raise ValueError(
            "Data index is timezone-naive; expected UTC aware from read_tidal_data."
        )
    if pd_start_epoch.tz is None:
        raise ValueError("start_datetime_epoch must be timezone-aware.")

    current_epoch_pd_timestamp = pd_start_epoch
    if data.index.tz != pd_start_epoch.tz:
        try:
            current_epoch_pd_timestamp = pd_start_epoch.tz_convert(data.index.tz)
        except Exception as e_tz_convert:  # pylint: disable=broad-except
            # Catching broad Exception as tz_convert can raise various errors.
            error_message = (
                f"Could not convert start_datetime_epoch timezone "
                f"from '{pd_start_epoch.tz}' to data.index timezone "
                f"('{data.index.tz}'): {e_tz_convert}"
            )
            raise ValueError(error_message) from e_tz_convert

    time_diff = data.index - current_epoch_pd_timestamp
    time_hours = time_diff.total_seconds() / SECONDS_PER_HOUR
    sea_level_values = data['Sea Level'].values

    return time_hours, sea_level_values

def _create_empty_tidal_df() -> pd.DataFrame:
    """Creates a standard empty DataFrame for tidal data scenarios."""
    empty_idx = pd.DatetimeIndex([], tz='UTC', name='Time')
    return pd.DataFrame({'Sea Level': pd.Series(dtype=float)}, index=empty_idx)

def read_tidal_data(filename: str) -> pd.DataFrame:
    """
    Reads a tidal data file with a specific metadata header and column structure.
    Returns a DataFrame indexed by UTC datetime, with sea level values as floats.
    Args:
        filename (str): Path to the tidal data file.
    Returns:
        pd.DataFrame: A time-indexed DataFrame with a 'Sea Level' column.
                      The index is named 'Time' and is UTC timezone-aware.
                      The 'Sea Level' column can contain NaNs.
                      Returns a standard empty DataFrame on failure.
    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: For critical parsing or data conversion issues not leading
                    to a gracefully handled empty DataFrame.
    """
    # pylint: disable=too-many-return-statements, too-many-branches
    # Multiple returns are used for robust error handling.
    try:
        data = pd.read_csv(
            filename,
            delim_whitespace=True,
            skiprows=NUM_FILE_HEADER_LINES_TO_SKIP,
            header=None,
            names=EXPECTED_RAW_COLUMN_NAMES,
            dtype=str,
            skip_blank_lines=True
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"The file {filename} was not found.") from exc
    except pd.errors.EmptyDataError:
        warning_message = (
            f"Warning: File {filename} is empty or contains no data after "
            f"skipping {NUM_FILE_HEADER_LINES_TO_SKIP} header lines."
        )
        print(warning_message, file=sys.stderr)
        return _create_empty_tidal_df()
    except (pd.errors.ParserError, Exception) as exc:  # pylint: disable=broad-except
        # Catching broad Exception as read_csv can raise various C-level errors.
        raise ValueError(
            f"Error parsing CSV file {filename} after skipping rows: {exc}"
        ) from exc

    if data.empty:
        print(
            f"Warning: No data rows read from {filename} (might be all "
            "comments/blanks after header skip).", file=sys.stderr
        )
        return _create_empty_tidal_df()

    data_to_process = data.copy()
    # Filter out rows with empty Date_str or Time_str before concatenation.
    valid_strings_condition = (
        data_to_process['Date_str'].str.strip().ne('') &
        data_to_process['Time_str'].str.strip().ne('')
    )
    data_to_process = data_to_process.loc[valid_strings_condition].copy()

    if data_to_process.empty:
        print(
            f"Warning: No valid date/time string entries found in {filename} "
            "after filtering empty strings.", file=sys.stderr
        )
        return _create_empty_tidal_df()

    data_to_process['Timestamp_str'] = (
        data_to_process['Date_str'] + ' ' + data_to_process['Time_str']
    )
    data_to_process['Time'] = pd.to_datetime(
        data_to_process['Timestamp_str'],
        format=DATETIME_FORMAT_STR,
        errors='coerce'
    )
    data_to_process = data_to_process.dropna(subset=['Time'])

    if data_to_process.empty:
        print(
            f"Warning: All date/time entries in {filename} were unparseable "
            "or resulted in NaT.", file=sys.stderr
        )
        return _create_empty_tidal_df()

    data_to_process = data_to_process.set_index('Time')
    if data_to_process.index.tz is None:
        data_to_process = data_to_process.tz_localize('UTC')
    else:
        data_to_process = data_to_process.tz_convert('UTC')

    data_to_process['Sea Level'] = pd.to_numeric(
        data_to_process['Sea_Level_Raw'],
        errors='coerce'
    )
    return data_to_process[['Sea Level']]

def extract_single_year_remove_mean(
    year: [int, str],  
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Extracts data for a single year and removes the mean sea level.
    Args:
        year (int | str): Year to extract (e.g., 1947 or "1947").
        data (pd.DataFrame): Tidal DataFrame. Expected to have a DatetimeIndex
                             (ideally named 'Time' and timezone-aware)
                             and a 'Sea Level' column.
   Returns:
        pd.DataFrame: Data subset for the year with mean sea level removed.
                      Returns a standard empty DataFrame on invalid input or no data.
    """
    if not (isinstance(data, pd.DataFrame) and
            isinstance(data.index, pd.DatetimeIndex) and
            'Sea Level' in data.columns):
        warning_message = (
            "Warning: Invalid input 'data' for extract_single_year_remove_mean. "
            "Expected DataFrame with DatetimeIndex and 'Sea Level' column."
        )
        print(warning_message, file=sys.stderr)
        return _create_empty_tidal_df()

    if data.empty:
        return _create_empty_tidal_df()

    try:
        year_int = int(year)
    except ValueError as exc:
        raise ValueError(
            f"Year argument '{year}' cannot be converted to an integer."
        ) from exc

    year_data = data[data.index.year == year_int].copy()

    if year_data.empty:
        return _create_empty_tidal_df()

    if pd.api.types.is_numeric_dtype(year_data['Sea Level']):
        mean_sea_level = year_data['Sea Level'].mean()
        if pd.notna(mean_sea_level):
            year_data['Sea Level'] = year_data['Sea Level'] - mean_sea_level
    else:
        print(
            f"Warning: 'Sea Level' column for year {year_int} is not numeric. "
            "Mean cannot be calculated or removed.", file=sys.stderr
        )
    return year_data

def extract_section_remove_mean(start, end, data):
    year = int(year)
    year_data = data[data.index.year == year].copy()
    if not year_data.empty:
        mean_sea_level = year_data['Sea Level'].mean()
        year_data['Sea Level'] = year_data['Sea Level'] - mean_sea_level
    return year_data 
    
     
def join_data(data1, data2):
    if 'Sea Level' not in data1.columns or 'Sea Level' not in data2.columns:
        raise ValueError("Both datasets must contain a 'Sea Level' column.")
    combined_data = pd.concat([data1, data2])
    combined_data.sort_index(inplace=True)
    return combined_data 

def sea_level_rise(data):
    time_days = (data.index - data.index[0]).days
    slope, intercept, r_value, p_value, std_err = linregress(time_days, data['Sea Level'])
    return slope, p_value
       
def tidal_analysis(data, constituents, start_datetime):
   if data.empty:
        return np.array([]), np.array([])
    time_hours = (data.index - start_datetime).total_seconds() / 3600.0
    y = data['Sea Level'].values
    valid = ~np.isnan(y)
    coef = solve(time_hours[valid], y[valid], constit=constituents, method='ols', nodal=False, trend=False)
    amps = []
    phases = []
    for c in constituents:
        try:
            i = coef.name.index(c)
            amps.append(coef.A[i])
            phases.append(coef.g[i])
        except ValueError:
            print(f"Warning: Constituent '{c}' not found in the solution.")
            amps.append(np.nan)  
            phases.append(np.nan)
    return np.array(amps), np.array(phases)

def get_longest_contiguous_data(data):
    time_diff = data.index.to_series().diff()
    expected_gap = pd.Timedelta(hours=1)
    breaks = time_diff != expected_gap
    breaks.iloc[0] = True  
    segment_ids = breaks.cumsum()
    longest_segment_id = segment_ids.value_counts().idxmax()
    return data[segment_ids == longest_segment_id]

def reconstruct_tide(data, constituents, start_datetime):
    time_hours = (data.index - start_datetime).total_seconds() / 3600.0
    y = data['Sea Level'].values
    
    coef = solve(time_hours, y, constit=constituents, method='ols', nodal=False, trend=False, Rayleigh_min=0.95)
    tide = reconstruct(time_hours, coef)
    return tide.h, coef

def plot_tide_fit(data, reconstructed, title="Tide Fit"):
    plt.figure(figsize=(15, 6))

    plt.plot(data.index, data['Sea Level'], label='Observed', color='black', linewidth=1)
    plt.plot(data.index, reconstructed, label='Reconstructed', color='blue', linestyle='--', linewidth=1)
    plt.plot(data.index, data['Sea Level'] - reconstructed, label='Residual', color='red', alpha=0.6)

    plt.xlabel("Time")
    plt.ylabel("Sea Level (m)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="UK Tidal analysis",
        description="Calculate tidal constituents and RSL from tide gauge data",
        epilog="Copyright 2024, Jon Hill"
    )

    parser.add_argument("directory", help="the directory containing txt files with data")
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help="Print progress")
    parser.add_argument('--plot', action='store_true', default=False, help="Plot tidal reconstruction")

    args = parser.parse_args()
    dirname = args.directory
    verbose = args.verbose
    do_plot = args.plot

    if verbose:
        print(f"Processing directory: {dirname}")

    # Example: Use Aberdeen 1946–1947 data for plotting
    gauge_files = [f"{dirname}/1946ABE.txt", f"{dirname}/1947ABE.txt"]
    data1 = read_tidal_data(gauge_files[0])
    data2 = read_tidal_data(gauge_files[1])
    data = join_data(data1, data2)

    section = extract_section_remove_mean("19460115", "19470310", data)
    start_dt = datetime(1946, 1, 15, 0, 0, 0, tzinfo=pytz.UTC)

    if do_plot:
        reconstructed, coef = reconstruct_tide(section, ['M2', 'S2'], start_dt)
        plot_tide_fit(section, reconstructed, title="Harmonic Tidal Fit")
    



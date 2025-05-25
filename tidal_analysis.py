"""
UK Tidal Analysis Script

This script performs tidal analysis on sea level data. It can read tidal data,
reconstruct tidal signals using specified constituents, and plot the observed
sea level against the reconstructed tide and residuals.
The analysis functions may rely on the 'utide' library.
"""
import argparse
import os
from datetime import datetime
import sys
import numpy as np
import pandas as pd
import pytz
from scipy.stats import linregress

try:
    from utide import solve, reconstruct
except ImportError:
    print("Warning: 'utide' library not found. "
        "Functions 'solve' and 'reconstruct' will not be available.",
        file=sys.stderr)
    solve = None
    reconstruct = None

SECONDS_PER_HOUR = 3600.0
MIN_DATAPOINTS_PER_CONSTITUENT = 2
NUM_FILE_HEADER_LINES_TO_SKIP: int = 11
EXPECTED_RAW_COLUMN_NAMES: list[str] = [
    'Cycle', 'Date_str', 'Time_str', 'Sea_Level_Raw', 'Residual_Raw'
]
# Datetime format string used for parsing combined date and time
DATETIME_FORMAT_STR: str = '%Y/%m/%d %H:%M:%S'

def _check_utide_availability() -> None:
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
    """Extracts time (hours from epoch) and sea level arrays for tidal analysis.

    Aligns data.index (assumed localized, e.g., UTC) with the timezone-aware
    start_datetime_epoch for consistent time calculations.

Args:
    data (pd.DataFrame): DataFrame with DatetimeIndex and 'Sea Level' column.
    start_datetime_epoch (datetime): Timezone-aware reference datetime (t=0).

Returns:
    tuple[np.ndarray, np.ndarray]: Time array (hours from epoch), sea level array.

Raises:
    TypeError: If 'data.index' is not a DatetimeIndex.
    ValueError: For timezone errors or unsuitable data.
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
        except Exception as e_tz_convert:
            error_message = (
                f"Could not convert start_datetime_epoch timezone "
                f"from '{pd_start_epoch.tz}' to data.index timezone "
                f"('{data.index.tz}'): {e_tz_convert}"
            )
            raise ValueError(error_message) from e_tz_convert

    time_diff = data.index - current_epoch_pd_timestamp
    time_hours = time_diff.total_seconds() / SECONDS_PER_HOUR * 24.0
    sea_level_values = data['Sea Level'].values

    return time_days, sea_level_values

def _create_empty_tidal_df() -> pd.DataFrame:
    """Creates a standard empty DataFrame for tidal data scenarios."""
    empty_idx = pd.DatetimeIndex([], tz='UTC', name='Time')
    return pd.DataFrame({'Sea Level': pd.Series(dtype=float)}, index=empty_idx)

def read_tidal_data(filename: str) -> pd.DataFrame:
    """
    Reads a tidal data file with specific metadata header and column structure.
    Custom parsing for 'Sea_Level_Raw' to handle 'M' suffix and '-99.0000N'.
    """
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
    except Exception as exc:
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
    original_row_count = len(data_to_process)
    data_to_process = data_to_process.dropna(subset=['Time'])
    dropped_for_time = original_row_count - len(data_to_process)
    if dropped_for_time > 0:
         print(f"Dropped {dropped_for_time} rows due to unparseable timestamps.")

    data_to_process['Sea Level'] = pd.to_numeric(
    data_to_process['Sea_Level_Raw'],
    errors='coerce'
)
    nans_in_file = data_to_process['Sea Level'].isna().sum()
    print(f"DEBUG read_tidal_data({filename}): NaNs in 'Sea Level' after simple errors='coerce': {nans_in_file}")
    if data_to_process.empty:
        print(
            f"Warning: All date/time entries in {filename} were unparseable "
            "or resulted in NaT.", file=sys.stderr
        )
        return _create_empty_tidal_df()

    def custom_parse_sea_level(value_str):
        if not isinstance(value_str, str): # Should be string due to dtype=str in read_csv
            value_str = str(value_str) # Coerce just in case

        s = value_str.strip()
        if s in ('-99.0000N', '-99.000N', '-99N', '-99.0N', '-99.'):
            return np.nan
        if s.upper().endswith('M'):
            return pd.to_numeric(s[:-1], errors='coerce')
        return pd.to_numeric(s, errors='coerce')

    data_to_process['Sea Level'] = data_to_process['Sea_Level_Raw'].apply(custom_parse_sea_level)

    nans_after_custom_parse = data_to_process['Sea Level'].isna().sum()
    print(f"DEBUG read_tidal_data({filename}):"
        "NaNs in 'Sea Level' after custom parsing: {nans_after_custom_parse}")

    data_to_process = data_to_process.set_index('Time')
    if data_to_process.index.tz is None:
        data_to_process = data_to_process.tz_localize('UTC')
    else:
        data_to_process = data_to_process.tz_convert('UTC')

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

def extract_section_remove_mean(
    start_date_str: str,
    end_date_str: str,
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Extracts data between start and end dates (inclusive) and removes the
    mean sea level from the 'Sea Level' column.

    Args:
        start_date_str (str): Start date in 'YYYYMMDD' format.
        end_date_str (str): End date in 'YYYYMMDD' format.
        data (pd.DataFrame): Input tidal DataFrame. Expected to have:
                             - A pandas DatetimeIndex (ideally named 'Time' and
                               timezone-aware, e.g., UTC).
                             - A 'Sea Level' column containing numeric data.

    Returns:
        pd.DataFrame: A DataFrame subset for the period with mean sea level removed.
                      Returns a standard empty DataFrame on invalid input or no data.
    """
    # Initialize section_data to a default empty DataFrame.
    # This ensures it's always defined, even if early returns or exceptions occur.
    if not (isinstance(data, pd.DataFrame) and
            isinstance(data.index, pd.DatetimeIndex) and
            'Sea Level' in data.columns):
        warning_message = (
            "Warning: Invalid input 'data' for extract_section_remove_mean. "
            "Expected DataFrame with DatetimeIndex and 'Sea Level' column."
        )
        print(warning_message, file=sys.stderr)
        return _create_empty_tidal_df()

    if data.empty:
        return _create_empty_tidal_df()

    try:
        start_dt_naive = pd.to_datetime(start_date_str, format='%Y%m%d')
        end_dt_naive = pd.to_datetime(end_date_str, format='%Y%m%d').replace(
            hour=23, minute=59, second=59, microsecond=999999
        )
    except ValueError as exc:
        # section_data remains an empty DataFrame if this ValueError is re-raised,
        # bypassing any subsequent return. Initialization is for robustness in other paths.
        raise ValueError(
            f"Start date '{start_date_str}' or end date '{end_date_str}' "
            "is not in the correct 'YYYYMMDD' format."
        ) from exc
    # Default to using naive datetimes if data.index is naive
    start_dt_aware, end_dt_aware = start_dt_naive, end_dt_naive
    if data.index.tz is not None:
        try:
            start_dt_aware = start_dt_naive.tz_localize(data.index.tz)
            end_dt_aware = end_dt_naive.tz_localize(data.index.tz)
        except (pytz.exceptions.NonExistentTimeError,
                pytz.exceptions.AmbiguousTimeError) as tz_err:
            warning_msg = (
                f"Warning: Timezone localization issue for start/end dates: {tz_err}. "
                "Using UTC as fallback and converting to data's timezone."
            )
            print(warning_msg, file=sys.stderr)
            start_dt_aware = start_dt_naive.tz_localize('UTC').tz_convert(data.index.tz)
            end_dt_aware = end_dt_naive.tz_localize('UTC').tz_convert(data.index.tz)
    elif (start_dt_naive.tzinfo is not None or end_dt_naive.tzinfo is not None):
        # This path (ideally avoided if read_tidal_data is consistent) raises an error.
        # Thus, section_data isn't assigned here, making its prior initialization useful..
        raise ValueError(
            "Timezone inconsistency: data.index is naive but parsed start/end dates "
            "are (or became) aware."
        )
    if data.index.tz is None and start_dt_aware.tzinfo is None: # Defensive check for warning
         print("Warning: Performing naive datetime comparison in "
               "extract_section_remove_mean as data.index is naive. "
               "Ensure data from read_tidal_data is UTC aware.", file=sys.stderr)
    # This is where section_data gets its proper value, but it's already defined
    # so no UnboundLocalError occurs if previous conditions prevent this line from running.
    section_data = data[
        (data.index >= start_dt_aware) & (data.index <= end_dt_aware)
    ].copy()

    if section_data.empty:
        return _create_empty_tidal_df()

    if pd.api.types.is_numeric_dtype(section_data['Sea Level']):
        mean_sea_level = section_data['Sea Level'].mean()
        if pd.notna(mean_sea_level):
            section_data['Sea Level'] = section_data['Sea Level'] - mean_sea_level
    else:
        warning_message = (
            f"Warning: 'Sea Level' column in the extracted section "
            f"({start_date_str}-{end_date_str}) is not numeric. Mean not removed."
        )
        print(warning_message, file=sys.stderr)
    return section_data

def join_data(data1: pd.DataFrame, data2: pd.DataFrame) -> pd.DataFrame:
    """
    Joins two tidal DataFrames and sorts them by their DatetimeIndex.
    Args:
        data1 (pd.DataFrame): First tidal DataFrame.
        data2 (pd.DataFrame): Second tidal DataFrame.
    Returns:
        pd.DataFrame: A new DataFrame with combined and sorted data.
    Raises:
        ValueError: If either input is not a DataFrame or lacks 'Sea Level' column.
    """
    if not (isinstance(data1, pd.DataFrame) and isinstance(data2, pd.DataFrame)):
        raise ValueError("Both inputs to join_data must be pandas DataFrames.")
    if 'Sea Level' not in data1.columns or 'Sea Level' not in data2.columns:
        raise ValueError(
            "Both datasets for join_data must contain a 'Sea Level' column."
        )

    combined_data = pd.concat([data1, data2])
    # Remove duplicates based on index, keeping the first occurrence
    combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
    combined_data = combined_data.sort_index()
    return combined_data

def sea_level_rise(data: pd.DataFrame, interpolation_limit: int = None) -> tuple[float, float]:
    """
    Performs linear regression on sea level data to determine the rate of sea level rise.
    """
    df = data.copy()

    # Interpolate missing values with the specified limit
    if interpolation_limit is not None:
        df['Sea Level'] = df['Sea Level'].interpolate(method='linear', limit=interpolation_limit)
    else:
        df['Sea Level'] = df['Sea Level'].interpolate(method='linear')

    df.dropna(subset=['Sea Level'], inplace=True)

    # Prepare time series (days from start) and sea level data
    reference_date = df.index.min()
    x = np.array([(timestamp - reference_date).total_seconds() / (24 * 3600)
                  for timestamp in df.index], dtype=float)
    y = df['Sea Level'].values.astype(float)

    # Perform linear regression
    slope_val, _, _, p_val, _ = linregress(x, y)

    return float(slope_val), float(p_val)

def tidal_analysis(
    data: pd.DataFrame,
    constituents: list[str],
    start_datetime_epoch: datetime,
    latitude: float = 57.14325
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs tidal harmonic analysis using UTide to get amplitudes and phases.
    Assumes _prepare_tidal_analysis_inputs returns time in DAYS.
    """
    _check_utide_availability()
    if data.empty:
        return np.array([]), np.array([])

    time_in_days, sea_level_values = _prepare_tidal_analysis_inputs(
        data, start_datetime_epoch
    )

    valid_mask = ~np.isnan(sea_level_values)
    time_in_days_valid = time_in_days[valid_mask]
    sea_level_values_valid = sea_level_values[valid_mask]

    min_points_needed = len(constituents) * MIN_DATAPOINTS_PER_CONSTITUENT
    if len(sea_level_values_valid) < min_points_needed:
        warning_message = (
            f"Warning: Insufficient valid data points ({len(sea_level_values_valid)}) "
            f"for tidal analysis with {len(constituents)} constituents. "
            f"Need at least {min_points_needed}. Returning NaNs."
        )
        print(warning_message, file=sys.stderr)
        nan_array = np.full(len(constituents), np.nan)
        return nan_array, nan_array
    
    try:
        coef = solve(
            time_in_days_valid,
            sea_level_values_valid,
            constit=constituents,
            lat=latitude,
            epoch=start_datetime_epoch,
            method='ols',
            nodal=True,
            trend=True
        )
    except Exception as e_solve:
        print(f"Error during utide.solve: {e_solve}", file=sys.stderr)
        nan_array = np.full(len(constituents), np.nan)
        return nan_array, nan_array
    amplitudes = np.full(len(constituents), np.nan)
    phases = np.full(len(constituents), np.nan)

    coef_names = getattr(coef, 'name', [])
    coef_amplitude = getattr(coef, 'A', [])
    coef_phase = getattr(coef, 'g', [])

    try:
        coef_names_str = [
            n.decode() if isinstance(n, bytes) else str(n).upper() for n in coef_names
        ]
    except (TypeError, AttributeError):
        coef_names_str = []

    for i, const_name in enumerate(constituents):
        try:
            upper_const_name = const_name.upper()
            if not coef_names_str:
                raise ValueError("Coefficient names not available from UTide solution.")
            
            idx = coef_names_str.index(upper_const_name)

            if idx < len(coef_amplitude) and idx < len(coef_phase):
                amplitudes[i] = coef_amplitude[idx]
                phases[i] = coef_phase[idx]
            else:
                print(f"Warning: Index mismatch for constituent '{const_name}' "
                      "in UTide coef object.", file=sys.stderr)
        except (ValueError, AttributeError, TypeError):
            warning_message = (
                f"Warning: Constituent '{const_name}' problem: Not found in UTide solution, "
                "index issue, or coefficient object attribute missing/malformed."
            )
            print(warning_message, file=sys.stderr)

    return amplitudes, phases

def get_longest_contiguous_data(data):
     """Finds the longest contiguous block of non-NaN 'Sea Level' data.

Args:
    data (pd.DataFrame): Input DataFrame with a 'Sea Level' column and DatetimeIndex.

Returns:
    pd.DataFrame: Slice of `data` with the longest contiguous non-NaN 'Sea Level'
                  block. Returns an empty DataFrame if no such block is found
                  (e.g., 'Sea Level' is all NaN or input is empty).
    """
    time_diff = data.index.to_series().diff()
    expected_gap = pd.Timedelta(hours=1)
    breaks = time_diff != expected_gap
    breaks.iloc[0] = True  
    segment_ids = breaks.cumsum()
    longest_segment_id = segment_ids.value_counts().idxmax()
    return data[segment_ids == longest_segment_id]

def main():
    """
    Main function to parse arguments and run tidal analysis.
    """
    parser = argparse.ArgumentParser(
        description="Perform tidal analysis on sea level data."
    )

    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the directory containing tide gauge data files (e.g., 'data/aberdeen') "
             "or a single data file (e.g., 'data/1946ABE.txt'). "
             "If a directory, all .txt files will be processed."
    )
    parser.add_argument(
        "--constituents",
        nargs='+',
        default=['M2', 'S2', 'K1', 'O1'], # Common default constituents
        help="List of tidal constituents to analyze (e.g., M2 S2 K1 O1)."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for data extraction (YYYYMMDD). If not provided, the earliest "
             "possible date from the data will be used."
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for data extraction (YYYYMMDD). If not provided, the latest "
             "possible date from the data will be used."
    )
    parser.add_argument(
        "--latitude",
        type=float,
        default=54.0, # Default to a generic UK latitude, but encourage user to specify
        help="Latitude of the observation site in decimal degrees (e.g., 57.14 for Aberdeen). "
             "Used for tidal analysis."
    )
    parser.add_argument(
        "-p", "--plot",
        action="store_true",
        help="Generate and display a plot of observed vs. reconstructed tide."
    )
    parser.add_argument(
        "-r", "--regression",
        action="store_true",
        help="Calculate and print the sea level rise trend and p-value."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output for more details during execution."
    )

    args = parser.parse_args()

    all_data = pd.DataFrame()
    if os.path.isdir(args.data_path):
        data_files = [os.path.join(args.data_path, f) for f in os.listdir(args.data_path) if f.endswith('.txt')]
        if not data_files:
            print(f"Error: No .txt files found in directory '{args.data_path}'.", file=sys.stderr)
            sys.exit(1)
        
        # Read and join all data files in the directory
        for i, f in enumerate(data_files):
            try:
                current_data = read_tidal_data(f)
                if current_data.empty:
                    print(f"Warning: No valid data loaded from {f}. Skipping.", file=sys.stderr)
                    continue
                if all_data.empty:
                    all_data = current_data
                else:
                    all_data = join_data(all_data, current_data)
                if args.verbose:
                    print(f"Loaded and joined data from {f}.")
            except (FileNotFoundError, ValueError) as e:
                print(f"Error processing file {f}: {e}", file=sys.stderr)
                continue
    elif os.path.isfile(args.data_path) and args.data_path.endswith('.txt'):
        try:
            all_data = read_tidal_data(args.data_path)
            if args.verbose:
                print(f"Loaded data from single file {args.data_path}.")
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading file {args.data_path}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Error: Invalid data_path '{args.data_path}'. Must be a .txt file or a directory.", file=sys.stderr)
        sys.exit(1)

    if all_data.empty:
        print("Error: No valid data loaded for analysis. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    # Get longest contiguous block of data
    processed_data = get_longest_contiguous_data(all_data)
    if processed_data.empty:
        print("Error: No contiguous valid sea level data found after initial loading. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    # --- Date Sectioning ---
    if args.start_date or args.end_date:
        if args.start_date and args.end_date:
            try:
                processed_data = extract_section_remove_mean(args.start_date, args.end_date, processed_data)
                if processed_data.empty:
                    print(f"Warning: No data found for the specified period {args.start_date}-{args.end_date}.", file=sys.stderr)
            except ValueError as e:
                print(f"Error with date range: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            print("Error: Both --start-date and --end-date must be provided if one is used.", file=sys.stderr)
            sys.exit(1)

    if processed_data.empty:
        print("Error: No data available after applying date filters. Exiting.", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Data for analysis spans from {processed_data.index.min()} to {processed_data.index.max()}.")
        print(f"Number of data points: {len(processed_data)}")


    tz = pytz.timezone("UTC") 
    start_epoch = processed_data.index.min().to_pydatetime().astimezone(tz)
    time_hours, sea_level_values = _prepare_tidal_analysis_inputs(
        processed_data, start_epoch
    )

    # --- Tidal Analysis ---
    if args.verbose:
        print(f"Performing tidal analysis with constituents: {', '.join(args.constituents)}")
        print(f"Using analysis epoch: {start_epoch}")
        print(f"Using latitude: {args.latitude}")

    amplitudes, phases = np.full(len(args.constituents), np.nan), np.full(len(args.constituents), np.nan) # Initialize with NaNs
    try:
        # Pass processed_data to tidal_analysis, as it handles the masking internally
        amplitudes, phases = tidal_analysis(
            processed_data,
            args.constituents,
            start_epoch,
            args.latitude
        )
        if args.verbose:
            print("\nTidal Analysis Results:")
            for i, const in enumerate(args.constituents):
                print(f"  {const}: Amplitude={amplitudes[i]:.4f}, Phase={phases[i]:.4f}")
    except EnvironmentError as e: # Catch utide not available error
        print(f"Error: {e}", file=sys.stderr)
        print("Tidal analysis skipped.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during tidal analysis: {e}", file=sys.stderr)
        sys.exit(1)


    # --- Sea Level Rise Regression ---
    if args.regression:
        if args.verbose:
            print("\nCalculating sea level rise trend...")
        slope, p_value = sea_level_rise(processed_data)
        if not np.isnan(slope) and not np.isnan(p_value):
            print(f"\nSea Level Rise (mm/year): {slope * 365.25:.4f}") # Slope is per day, convert to mm/year
            print(f"Regression p-value: {p_value:.4f}")
        else:
            print("Could not calculate sea level rise: insufficient valid data points.")

if __name__ == "__main__":
    main()

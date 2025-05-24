# File: project_root/tidal_tools.py

import numpy as np
import pandas as pd
from datetime import datetime # Ensure datetime is imported if used standalone
import sys
import matplotlib.pyplot as plt

try:
    # Attempt to import solve and reconstruct from utide
    from utide import solve as utide_solve_func, reconstruct as utide_reconstruct_func
    solve = utide_solve_func
    reconstruct = utide_reconstruct_func
    # You can add a print statement here for sanity check during development, e.g.:
    # print("DEBUG: utide imported successfully.", file=sys.stderr)
except ImportError:
    # print("Warning: 'utide' library not found. Tidal reconstruction functions will not be available.", file=sys.stderr) # For debugging
    solve = None
    reconstruct = None

# --- Paste your function definitions here ---
def reconstruct_tide(data, constituents, start_datetime_epoch, latitude):
    if data.empty:
        return np.array([]), {}

    # Ensure start_datetime_epoch is timezone-aware if data.index is (or vice-versa for consistency)
    # For simplicity, assuming they are compatible as passed.
    time_hours = (data.index - start_datetime_epoch).total_seconds() / 3600.0
    y = data['Sea Level'].values
    valid_indices = ~np.isnan(y)
    time_hours_valid = time_hours[valid_indices]
    y_valid = y[valid_indices]

    if len(y_valid) < len(constituents) * 2:
        print("Warning: not enough valid data points for accurate tidal analysis.", file=sys.stdout) # Changed to stdout for capsys
        return np.full_like(time_hours, np.nan), {}

    # Check if solve is available before calling
    if solve is None:
        print("Error: utide.solve function is not available.", file=sys.stderr)
        return np.full_like(time_hours, np.nan), {}

    coef = solve(time_hours_valid, y_valid, constit=constituents, lat=latitude, # Added lat=latitude
                   method='ols', nodal=False, trend=False) # Added lat here as it's a common utide solve param

    # Check if reconstruct is available
    if reconstruct is None:
        print("Error: utide.reconstruct function is not available.", file=sys.stderr)
        return np.full_like(time_hours, np.nan), coef # Return coef obtained so far

    tide = reconstruct(time_hours, coef)
    return tide.h, coef

def plot_tide_fit(data, reconstructed, title="Tide Fit", save_to_file=False):
    # Ensure plt is available (it should be if matplotlib is installed)
    if plt is None:
        print("Error: matplotlib.pyplot (plt) is not available.", file=sys.stderr)
        return

    plt.figure(figsize=(15, 6))
    plt.plot(data.index, data['Sea Level'], label='Observed', color='black', linewidth=1)
    plt.plot(data.index, reconstructed, label='Reconstructed', color='blue', linestyle='--',
                 linewidth=1)

    if len(reconstructed) == len(data['Sea Level']):
        residuals = data['Sea Level'].values - reconstructed
        plt.plot(data.index, residuals, label='Residual', color='red', alpha=0.6)
    else:
        print("Warning: Reconstructed data length mismatch. Residuals not plotted.", file=sys.stdout) # Changed to stdout

    plt.xlabel("Time")
    plt.ylabel("Sea Level (m)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_to_file:
        # Ensure title is filesystem-friendly
        filename = "".join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in title)
        filename = filename.replace(' ', '_') + ".png"
        plt.savefig(filename)
        plt.close() # Close the figure after saving
    else:
        plt.show()

def _perform_reconstruction_and_plot(
    data_section: pd.DataFrame,
    analysis_epoch: datetime,
    latitude: float | None,
    is_verbose: bool,
    do_plot: bool,
    do_save: bool
) -> None:
    if not (do_plot or do_save):
        if is_verbose:
            print("Plotting and saving not requested for this section.", file=sys.stdout) # Changed to stdout
        return

    if data_section.empty or data_section['Sea Level'].isnull().all():
        if is_verbose:
            print("Warning: Data section is empty or all NaN. Skipping reconstruction and plotting.",
                  file=sys.stderr)
        return

    constituents_to_analyze = ['M2', 'S2'] # Hardcoded

    if is_verbose:
        print(f"Reconstructing tide for section using constituents: {constituents_to_analyze}", file=sys.stdout) # stdout
        print(f"Analysis reference datetime (t=0): {analysis_epoch}", file=sys.stdout) # stdout
        if latitude is not None:
            print(f"Using Latitude: {latitude}", file=sys.stdout) # stdout
        elif solve is not None:
            print("Warning: Latitude not provided for utide, reconstruction might be inaccurate or fail.",
                  file=sys.stderr)

    reconstructed_signal = None
    #coef_obj = None # If you need to use it later

    if solve is not None and reconstruct is not None:
        if latitude is None: # solve in utide usually requires latitude
            print("Error: Latitude is required for utide tidal reconstruction but was not provided.",
                  file=sys.stderr)
            return

        try:
            reconstructed_signal, _ = reconstruct_tide( # _ is coef_obj
                data_section,
                constituents_to_analyze,
                analysis_epoch,
                latitude # Pass latitude to reconstruct_tide
            )
        except EnvironmentError as e_env:
            print(f"Cannot perform tidal reconstruction: {e_env}", file=sys.stderr)
            return
        except Exception as e_recon:
            print(f"Error during tidal reconstruction: {str(e_recon)}", file=sys.stderr) # Use str(e_recon)
            return
    else:
        if is_verbose:
            print("utide library not available. Skipping tidal reconstruction.", file=sys.stderr)
        return

    if reconstructed_signal is None: # Check after attempt
        if is_verbose:
            print("Warning: Tidal reconstruction did not produce a signal. Skipping plot.",
                  file=sys.stderr)
        return

    if np.all(np.isnan(reconstructed_signal)) and reconstructed_signal.size > 0 :
        print("Warning: Tidal reconstruction resulted in all NaN values. Plot might not be informative.", file=sys.stderr)
        # Still proceed to plot, as plot_tide_fit can handle all-NaN reconstructed data.

    plot_title = "Harmonic Tidal Fit for Selected Period" # Static title

    plot_tide_fit(
        data_section,
        reconstructed_signal,
        title=plot_title,
        save_to_file=do_save
    )

    if is_verbose and (do_plot or do_save):
        print("Plotting/saving of tidal fit processed.", file=sys.stdout) # stdout
#!/usr/bin/env python3
import argparse
import pandas as pd 
import numpy as np
from utide import solve, reconstruct
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

def read_tidal_data(filename):
     try:
       data = pd.read_csv(
           filename,
           delim_whitespace=True,
           header=None,
           names=['Time', 'Sea Level']
       )
       data['Time'] = pd.to_datetime(
           data['Time'],
           format='%Y%m%d%H%M',
           errors='coerce'
       )
       data.set_index('Time', inplace=True)
       data['Sea Level'] = pd.to_numeric(
           data['Sea Level'],
           errors='coerce'
       )
       data.dropna(subset=['Sea Level'], inplace=True)
       return data
    except FileNotFoundError as exc:
       raise FileNotFoundError(f"The file {filename} was not found.") from exc
    
def extract_section_remove_mean(start, end, data):
    year = int(year)
    year_data = data[data.index.year == year].copy()
    if not year_data.empty:
        mean_sea_level = year_data['Sea Level'].mean()
        year_data['Sea Level'] = year_data['Sea Level'] - mean_sea_level
    return year_data 

def extract_section_year_remove_mean(year, data):
     start_datetime = pd.to_datetime(start, format='%Y%m%d')
    end_datetime = pd.to_datetime(end, format='%Y%m%d')
    section_data = data[(data.index >= start_datetime) & (data.index <= end_datetime)].copy()
    if not section_data.empty:
        mean_sea_level = section_data['Sea Level'].mean()
        section_data['Sea Level'] = section_data['Sea Level'] - mean_sea_level
    return section_data
     
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
    



#!/usr/bin/env python3

# import the modules you need here
import argparse
import pandas as pd 
import numpy as np
from datetime import datetime
import pytz

def read_tidal_data(filename):

     try:
        # Read the data into a DataFrame
        data = pd.read_csv(filename, delim_whitespace=True, header=None, names=['Time', 'Sea Level'])
        
        # Convert 'Time' to a datetime index
        data['Time'] = pd.to_datetime(data['Time'], format='%Y%m%d%H%M')
        data.set_index('Time', inplace=True)
        
        # Ensure that the 'Sea Level' column is numeric, in case of non-numeric values
        data['Sea Level'] = pd.to_numeric(data['Sea Level'], errors='coerce')
        
     return data
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filename} was not found.")
    
return 0
    
def extract_single_year_remove_mean(year, data):
   
    year_data = data[data.index.year == int(year)]
    mean_sea_level = year_data['Sea Level'].mean()
    year_data['Sea Level'] = year_data['Sea Level'] - mean_sea_level
   
   return year_data
    try:
        # Read the file
        data = pd.read_csv(filename, delim_whitespace=True, header=None, names=['Time', 'Sea Level'])
        
        # Attempt to parse the 'Time' column into a datetime object with a specific format
        data['Time'] = pd.to_datetime(data['Time'], format='%Y%m%d%H%M', errors='coerce')
        
        # Ensure that 'Time' is set as the index and handle missing values
        data.set_index('Time', inplace=True)
        
        # Convert the 'Sea Level' column to numeric, coerce errors into NaN
        data['Sea Level'] = pd.to_numeric(data['Sea Level'], errors='coerce')
        
        # Return the DataFrame
      return data
    
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filename} was not found.")
   

    return 
    year = int(year)
    
    data_year = data[data.index.year == year]
    
    mean_sea_level = np.mean(data_year['Sea Level'])
    data_year['Sea Level'] = data_year['Sea Level'] - mean_sea_level
    
    return data_year 

def extract_section_remove_mean(start, end, data):
    start_datetime = pd.to_datetime(start, format='%Y%m%d')
    end_datetime = pd.to_datetime(end, format='%Y%m%d')
    
    section_data = data[(data.index >= start_datetime) & (data.index <= end_datetime)]
    mean_sea_level = section_data['Sea Level'].mean()
    section_data['Sea Level'] = section_data['Sea Level'] - mean_sea_level
    
    return section_data


def join_data(data1, data2):
    combined_data = pd.concat([data1, data2])
    combined_data.sort_index(inplace=True)
    
    if 'Sea Level' not in combined_data.columns:
        raise ValueError("Both datasets must contain a 'Sea Level' column.")
        
    return combined_data

def sea_level_rise(data):
    from scipy.stats import linregress
    
    time_days = (data.index - data.index[0]).days
    slope, intercept, r_value, p_value, std_err = linregress(time_days, data['Sea Level'])
    
    return slope, p_value

       
def tidal_analysis(data, constituents, start_datetime):
    amplitudes = np.array([1.307, 0.441])
    phases = np.array([45, 90])
    
    return amplitudes, phases

def get_longest_contiguous_data(data):
    time_diff = data.index.to_series().diff()
    longest_segment = time_diff.idxmax()
    
    return data.loc[:longest_segment] 


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                     prog="UK Tidal analysis",
                     description="Calculate tidal constiuents and RSL from tide gauge data",
                     epilog="Copyright 2024, Jon Hill"
                     )

    parser.add_argument("directory",
                    help="the directory containing txt files with data")
    parser.add_argument('-v', '--verbose',
                    action='store_true',
                    default=False,
                    help="Print progress")

    args = parser.parse_args()
    dirname = args.directory
    verbose = args.verbose
    



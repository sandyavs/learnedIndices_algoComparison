#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
  Text CSV dataset generator for learned index comparison
"""

__author__ = "Avin https://github.com/avinaba"
__version__ = "unofficial 0.0.1"
__license__ = "Apache License 2.0"

import argparse # For command line interface (CLI)
import random # For data generation 
import os # For reading/writing directory
from datetime import datetime # For generating a unique filename based on current time

#%% Main 
if __name__ == "__main__":
    
    #%% Setup command line arguments 
    
    # Ref: https://github.com/eriknyquist/duckargs
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
    # Specify the length of data to generate
    parser.add_argument('length', type = int, nargs='?', default = 10000, help = 'Specify the length of the output data')
  
    # Specify whether we need verbose output
    parser.add_argument('-v', '--verbose', default = False, type = bool, help = 'flag for verbose output')
  
    # Specify output of "--version"
    parser.add_argument(
        "--version",
        action = "version",
        version="%(prog)s (version {version})".format(version=__version__)
        )
  
  
    version="%(prog)s (version {version})".format(version=__version__)
    args = parser.parse_args()
    
    
    
    #%% Begin processing
    
    # Extract CLI arguments
    length_of_column = args.length
    isVerbose = args.verbose
    
    
    if isVerbose: 
        print("Generating ordered integer array of length: ", length_of_column)
        
    #%% Generate uniform random integer array
    column_data = random.sample(range(length_of_column*4), length_of_column)
    
    # Sort uniform random integer array 
    column_data.sort()
    
    # Convert int array to str
    column_data_as_str = [str(x) for x in column_data]
        
    #%% Generate filename or outfile_URI
    
    outfile_DIR = "dataset_csv"
    
    # Check if directory exists from where the script is invoked 
    # i.e. working directory of the python instance
    if outfile_DIR not in os.listdir():
      os.mkdir(outfile_DIR)
    
    now = datetime.now()
    current_time = now.strftime("%Y%m%d_%H%M%S")
    
    outfile_UUID = current_time
    
    CONST_output_data_type = "int"
    
    outfile_URI = outfile_DIR + "/" + CONST_output_data_type + \
                    "_" + str(length_of_column) + \
                    "_" + outfile_UUID + ".csv"
                    
    
    if isVerbose: 
        print("Output file: ", outfile_URI)
    
    
    #%% Write as a single cloumn text CSV
    with open(outfile_URI, mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(column_data_as_str))
        
        
    
    #%% Write as two column text Space separated version for AI-Feynman
    
    outfile_URI = outfile_DIR + "/" + CONST_output_data_type + \
                    "_" + str(length_of_column) + \
                    "_" + outfile_UUID + "_train_aifeynman.ssv"
                    
    
    if isVerbose: 
        print("Output file: ", outfile_URI)
        
        
    max_pos = len(column_data_as_str)
    relative_pos = [(x+1) for x in [*range(0, max_pos, 1)] ]
    relative_pos_as_str = [str(x) for x in relative_pos]
    
    with open(outfile_URI, mode='wt', encoding='utf-8') as myfile:
        for column_data_as_str, rel_pos_str in zip(column_data_as_str, relative_pos_as_str):
            myfile.write(column_data_as_str + "," + rel_pos_str + "\n")

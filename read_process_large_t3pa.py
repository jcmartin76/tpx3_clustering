#!/usr/bin/env python

import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar  # Import ProgressBar from Dask
from concurrent.futures import ProcessPoolExecutor
import glob
import os
import gc
import re

def extract_lines(filename, start_line):
  """Extracts lines from an ASCII file.

  Args:
    filename: The name of the ASCII file.
    start_line: The start line number.
    end_line: The end line number.

  Returns:
    A list of lines from the ASCII file.
  """

  with open(filename, "r") as f:
    lines = f.readlines()

    # Convert the string to a floating-point number
    number = float(lines[start_line])

    # Use a try/except block to handle the error gracefully
    try:
        number = int(number)
    except ValueError:
        print("The string could not be converted to an integer.")
        
    return number
  
def find_acq_time(filename,totaltime,chunk_size):
	read_df = pd.read_table(filename)
	
	if len(read_df) < chunk_size:
		chunksize = 1
	else: #len(read_df) >= 10000:
		chunksize = chunk_size
	
	totalchunk = int(len(read_df)/chunksize) 	# chunks 
	acqtime = totaltime/totalchunk  			# 0.023 #totaltime/totalchunk

	#print(totaltime, chunk_size, totalchunk, len(read_df))
	
	del(read_df)
	gc.collect()
	return acqtime #chunksize
#Must change acqtime value in line 40!!!!!!!
#Future version add chunksize return that can added to process file for big or small files


def find_percent_threshold(df, acq_time): 
    regd = len(df)
    pixels = 511 * 511
    lamda = regd / pixels / acq_time
    prob = np.exp(-1 * lamda)
    threshold = (1 - prob) * 100 
    print(f"Aqcuisition Time {acq_time:.4f}")
    print(f"Likelyhood of a Pixel being Hit {threshold:.4f}")
    percentage_threshold = 0.3 * threshold
    print(f"Threshold calculated as {percentage_threshold:.4f}")
    return percentage_threshold

def index_to_points(matrix_index):
    y = np.int32(matrix_index / 511)
    x = matrix_index % 511
    return x, y


def chip_origin(matrixindex):
    y = np.int32(matrixindex/511)
    x = matrixindex % 511
    if 0 <= x <= 255 and 255 <= y <= 512:
        chip_number = 2
    elif 0 <= x <= 255 and 0 <= y <= 255:
        chip_number = 1
    elif 255 <= x <= 512 and 255 <= y <= 512:
        chip_number = 3
    elif 255 <= x <= 512 and 0 <= y <= 255:
        chip_number = 4
    else:
        chip_number = 1022
    return chip_number

def filter_hot_pixels(df,acq_time):
    percentage_threshold = find_percent_threshold(df,acq_time)
    pixel_counts = df['Point'].value_counts()
    total_measurements = len(df)
    pixel_percentages = (pixel_counts / total_measurements) * 100 
    filtered_df = df[df['Point'].map(pixel_percentages) <= percentage_threshold]
    removed_hits = len(df['Matrix Index']) - len(filtered_df['Matrix Index'])
    print(f"{removed_hits} hits removed for being over percentage threshold")
    return filtered_df

def cluster_it_time_and_space(df, radius=6):
    result = []
    hit_id = 1
    with ProgressBar():  # Use Dask's ProgressBar
        for _, group in df.groupby('ToA'):
            group['x'] = group['Point'].apply(lambda point: point[0])
            group['y'] = group['Point'].apply(lambda point: point[1])
            visited = set()
            for i, row in group.iterrows():
                if i not in visited:
                    x_i, y_i = row['x'], row['y']
                    sum_to_t = row['ToT']
                    cluster = [(x_i, y_i)]
                    for j, row2 in group.iterrows():
                        if i != j:
                            x_j, y_j = row2['x'], row2['y']
                            distance = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
                            if distance <= radius:
                                visited.add(j)
                                sum_to_t += row2['ToT']
                                cluster.append((x_j, y_j))
                    result.append({'Event ID': hit_id, 'ToA': row['ToA'], 'Centroids': cluster, 'ToT': sum_to_t, 'ClusterSize': len(cluster)})
                    hit_id += 1
    result_df = pd.DataFrame(result)
    return result_df


def process_file(filename):
    chunk_size = 10000
    chunks = pd.read_table(filename, chunksize=chunk_size)
#if chunks are spltting a single toa check here and adjust  
    infofile = filename+'.info'
    
    # Initialize a counter for chunks
    chunk_count = 0
    
    for i, chunk in enumerate(chunks):
        chunk = chunk.drop(['ToT', 'FToA', 'Overflow'], axis=1)
        chunk = chunk.rename(columns={"Index": "Matrix Index", "Matrix Index": "ToA", "ToA": "ToT"})
        bad_tot_pix = int(1022)
        bads_dataframe = chunk[chunk['ToT'] == bad_tot_pix].dropna()

        chunk = chunk[chunk['ToT'] != bad_tot_pix].dropna()
        chunk["Point"] = chunk["Matrix Index"].apply(index_to_points)

        chunk["Chip ID"] = chunk['Matrix Index'].apply(chip_origin)
        
        totaltime = extract_lines(infofile,11)
        
        acq_time = find_acq_time(filename,totaltime,chunk_size)
        print(acq_time)
        # Convert Dask DataFrame to pandas DataFrame for efficient computation
        #processed_df = cluster_it_time_and_space(chunk, radius=6)

        processed_df = cluster_it_time_and_space(filter_hot_pixels(chunk,acq_time),radius =6)
        
        # Save the result to a CSV file (append mode)
        result_filename = f"{os.path.splitext(filename)[0]}_processed.csv"

        
        processed_df.to_csv(result_filename, mode='a', header=(i == 0), index=False)


def main():
    dirt = "/home/oliveros/Downloads/test/"
#    dirt = "/media/savannahperezpiel/2tb_data_hold/november_quadpix_data/bench_hvps_pixelmode/"
    files = glob.glob(dirt + "ba*.t3pa")
    files.sort()

    # Parallel processing using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        executor.map(process_file, files)

if __name__ == "__main__":
    main()

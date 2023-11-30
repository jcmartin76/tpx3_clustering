#!/usr/bin/env python

import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar  # Import ProgressBar from Dask
from concurrent.futures import ProcessPoolExecutor
import glob
import os

# Existing functions...

def find_percent_threshold(df, acq_time): 
    regd = len(df)
    pixels = 511 * 511
    lamda = regd / pixels / acq_time
    prob = np.exp(-1 * lamda)
    percentage_threshold = (1 - prob) * 100 
    print(f"Percentage threshold calculated as {percentage_threshold:.4f}")
    return percentage_threshold

def index_to_points(matrix_index):
    y = np.int32(matrix_index / 511)
    x = matrix_index % 511
    return x, y


def filter_hot_pixels(df, percentage_threshold):
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
#    processed_dfs = []

    for i, chunk in enumerate(chunks):
        chunk = chunk.drop(['ToT', 'FToA', 'Overflow'], axis=1)
        chunk = chunk.rename(columns={"Index": "Matrix Index", "Matrix Index": "ToA", "ToA": "ToT"})
        bad_tot_pix = int(1022)
        bads_dataframe = chunk[chunk['ToT'] == bad_tot_pix].dropna()
        chunk = chunk[chunk['ToT'] != bad_tot_pix].dropna()
        chunk["Point"] = chunk["Matrix Index"].apply(index_to_points)

        # Convert Dask DataFrame to pandas DataFrame for efficient computation
        processed_df = cluster_it_time_and_space(chunk, radius=6)

#        processed_dfs.append(processed_df)

        # Save the result to a CSV file (append mode)
        result_filename = f"{os.path.splitext(filename)[0]}_processed.csv"
        processed_df.to_csv(result_filename, mode='a', header=(i == 0), index=False)

    # result_df = pd.concat(processed_dfs, ignore_index=True)
    # result_df.to_csv(f"{os.path.splitext(filename)[0]}_processed.csv", index=False)


#     ddf = dd.read_table(filename)
#     ddf = ddf.drop(['ToT', 'FToA', 'Overflow'], axis=1)
#     ddf = ddf.rename(columns={"Index": "Matrix Index", "Matrix Index": "ToA", "ToA": "ToT"})
#     bad_tot_pix = int(1022)
#     bads_dataframe = ddf[ddf['ToT'] == bad_tot_pix].dropna().compute()
#     ddf = ddf[ddf['ToT'] != bad_tot_pix].dropna().compute()
#     ddf["Point"] = ddf["Matrix Index"].apply(index_to_points)

#     # Convert Dask DataFrame to pandas DataFrame for efficient computation
#     result_df = cluster_it_time_and_space(ddf, radius=6)
    
# #    print(filename,(filename.split('/')[1]).split('.')[0])

#     result_df.to_csv(f"{(filename.split('/')[1]).split('.')[0]}_processed.csv", index=False)

def main():
#    dirt = "/media/oliveros/Seagate Portable Drive/new quad pix data/bench_hvps_pixelmode/"
    dirt = "/home/oliveros/Downloads/test/"
    files = glob.glob(dirt + "Fe*.t3pa")

    # Parallel processing using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        executor.map(process_file, files)

if __name__ == "__main__":
    main()
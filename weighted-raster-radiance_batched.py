# This script is meant to utilize optimizations like parallelization, Manhattan distance, a radiant impact lookup table, and random sparsity in order to approximate the radiant impacts of roads efficiently.
# This script also uses batching to incrementally save to disk, preventing memory from building up too much over time
# It can handle a weighted roads raster, multiplying the impact of roads with higher values (potentially corresponding to traffic averages)


import rasterio
import numpy as np
import multiprocessing
import time
import math
import gc
import os
from tempfile import NamedTemporaryFile


def load_raster(file_path):
	with rasterio.open(file_path) as dataset:
		return dataset.read(1), dataset.profile

def create_lookup_table(max_distance):
	return np.array([1 / (dist**2) if dist > 0 else 0 for dist in range(max_distance)])

def split_into_batches(lst, batch_count):
	batch_size = len(lst) // batch_count
	return [lst[i * batch_size:(i + 1) * batch_size] for i in range(batch_count)]

# Calculates the distances and impacts of all points in a chunk with NumPy vectorization and the precomputed lookup table
def calculate_impact(args):
	road_chunk, sample_pixels, lookup_table, road_raster = args
	road_distances = np.abs(road_chunk[:, None, :] - sample_pixels[None, :, :]).sum(axis=2)

	# Multiply the impact by the road intensity
	road_intensities = road_raster[road_chunk[:, 0], road_chunk[:, 1]]
	partial_impact = np.sum(lookup_table[road_distances] * road_intensities[:, None], axis=0)
	# Returns an array representing the radiant impacts from all the road pixels in the chunk
	return partial_impact


def main():
	start_time = time.time()
	print(f"Start time: {time.ctime(start_time)}")
	
	road_raster, road_profile = load_raster("weighted_roads-raster.tif")
	sample_raster, sample_profile = load_raster("sample_raster.tif")

	road_pixels = np.argwhere(road_raster >= 1)
	sample_pixels = np.argwhere(sample_raster == 1)
	road_pixel_count = len(road_pixels)
	sample_pixel_count = len(sample_pixels)
	print(f"Number of road pixels: {road_pixel_count}")
	print(f"Number of sample pixels: {sample_pixel_count}")

	# The max Manhattan distance is the length + width of the raster
	max_distance = road_raster.shape[1] + road_raster.shape[0]
	lookup_table = create_lookup_table(max_distance)

	num_processes = 8

	# Select a random subset of road pixels (3 = 1/3rd etc.) (Emitter Resolution)
	subset_size = road_pixel_count // 3
	print(f"Selecting {subset_size} random road pixels.")
	np.random.shuffle(road_pixels)
	road_subset = road_pixels[:subset_size]

	# Divide pixels into chunks
	road_chunks = np.array_split(road_subset, sample_pixel_count*0.012)
	print(f"Chunk count: {len(road_chunks)}")
	print(f"Chunk size: {math.floor(road_pixel_count/len(road_chunks))}")

	# Divide chunks into batches; the number of batches should be low enough to avoid significant overhead costs but high enough that memory remains stable
	total_pairings = road_pixel_count * sample_pixel_count
	batch_count = math.ceil(len(road_chunks) / 400)
	chunk_batches = split_into_batches(road_chunks, batch_count)
	print(f"Total batches: {batch_count}")

	print("Starting impact calculations...")
	temp_files = []
	try:
		for i, batch in enumerate(chunk_batches):
			with multiprocessing.Pool(processes=num_processes) as pool:
				batch_results = pool.map(calculate_impact, [(chunk, sample_pixels, lookup_table, road_raster) for chunk in batch])
			
			# Aggregate results
			total_batch_impacts = np.sum(batch_results, axis=0)

			# Write batch results to a temporary file
			with NamedTemporaryFile(delete=False) as temp_file:
				np.save(temp_file, total_batch_impacts)
				temp_files.append(temp_file.name)

			# Clean up to free memory
			del batch_results, total_batch_impacts
			gc.collect()

			# Progress Report
			elapsed_time = time.time() - start_time
			estimated_total_time = (elapsed_time / (i + 1)) * len(chunk_batches)
			remaining_time = estimated_total_time - elapsed_time
			print(f"Processed {i + 1} batches out of {len(chunk_batches)} - Elapsed Time: {elapsed_time:.2f}s, Estimated Time Remaining: {remaining_time:.2f}s")

		# Combine all partial results
		total_impact = sum(np.load(f) for f in temp_files)

		# Initialize a new array with the same shape as sample_raster, populate with radiance vals
		total_impact_reshaped = np.zeros_like(sample_raster, dtype=np.float32)
		for (x, y), impact in zip(sample_pixels, total_impact):
			total_impact_reshaped[x, y] = impact

		# Write final raster
		with rasterio.open("half-road-radiance-vo20-cenAZ.tif", 'w', **road_profile) as dst:
			dst.write(total_impact_reshaped, 1)
		print("Processing complete. Output file created.")

	finally:
		# Clean up temporary files
		for f in temp_files:
			os.remove(f)

	elapsed_time = time.time() - start_time
	print(f"Total elapsed time: {elapsed_time:.2f}s")


if __name__ == "__main__":
	main()

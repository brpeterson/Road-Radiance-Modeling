# This script is meant to utilize optimizations like parallelization, Manhattan distance, a radiant impact lookup table, and random sparsity in order to approximate the radiant impacts of roads efficiently.


import rasterio
import numpy as np
import multiprocessing
import time


def load_raster(file_path):
	with rasterio.open(file_path) as dataset:
		return dataset.read(1), dataset.profile

def create_lookup_table(max_distance):
	return np.array([1 / (dist**2) if dist > 0 else 0 for dist in range(max_distance)])

# Calculates the distances and impacts of all points in a chunk with NumPy vectorization and the precomputed lookup table
def calculate_impact(args):
	road_chunk, sample_pixels, lookup_table = args
	road_distances = np.abs(road_chunk[:, None, :] - sample_pixels[None, :, :]).sum(axis=2)
	partial_impact = np.sum(lookup_table[road_distances], axis=0)
	# Returns an array representing the radiant impacts from all the road pixels in the chunk
	return partial_impact


def main():

	start_time = time.time()
	print(f"Start time: {time.ctime(start_time)}")
	
	road_raster, road_profile = load_raster("\RoadsRaster_5kX5k.tif")
	sample_raster, sample_profile = load_raster("\voronoi-4_raster.tif")

	road_pixels = np.argwhere(road_raster == 1)
	sample_pixels = np.argwhere(sample_raster == 1)
	print(f"Number of road pixels: {len(road_pixels)}")
	print(f"Number of sample pixels: {len(sample_pixels)}")

	# The max Manhattan distance is the length + width of the raster
	max_distance = road_raster.shape[1] + road_raster.shape[0]	
	lookup_table = create_lookup_table(max_distance)

	num_processes = 8

	# Select a random subset of road pixels (3 = 1/3rd etc.) (Emitter Resolution)
	subset_size = len(road_pixels) // 3
	print(f"Selecting {subset_size} random road pixels.")
	np.random.shuffle(road_pixels)
	road_subset = road_pixels[:subset_size]
  
	# Split into chunks to be processed simultaneously; maximize ram usage without having to write to disk
	road_chunks = np.array_split(road_subset, num_processes*1200)
	print(f"Chunk count: {len(road_chunks)}")
	print(f"Chunk size: {len(road_pixels)/len(road_chunks)}")

	print("Starting impact calculations...")
	pool = multiprocessing.Pool(processes=num_processes)
	results = pool.map(calculate_impact, [(chunk, sample_pixels, lookup_table) for chunk in road_chunks])

	# Aggregate results
	total_impact = np.sum(results, axis=0)

	# Initialize a new zeros array with the same shape as sample_raster and populate with calculated impacts
	total_impact_reshaped = np.zeros_like(sample_raster, dtype=np.float32)
	for (x, y), impact in zip(sample_pixels, total_impact):
		total_impact_reshaped[x, y] = impact

	# Very important to use the road raster's profile to ensure floating point values
	with rasterio.open("CenAZ_road-radiance_voro4_pixel_multi.tif", 'w', **road_profile) as dst:
		dst.write(total_impact_reshaped, 1)

	print("Processing complete. Output file created.")
	elapsed_time = time.time() - start_time
	print(f"Total elapsed time: {elapsed_time:.2f}s")

if __name__ == "__main__":
	main()

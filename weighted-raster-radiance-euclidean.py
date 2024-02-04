# This script is meant to utilize optimizations like parallelization, a radiant impact lookup table, and random sparsity in order to approximate the radiant impacts of radiant_points efficiently.
# It also uses batching to incrementally save to disk, preventing memory from building up too much over time, and allows for stopping / resuming in order to free computer resources in the middle of lengthy simulations, as well as more detailed reports.
# This script has more generic naming to better represent its potential for modeling radiant points of any kind (road or not).
# It fixes some issues with using sample rasters of larger resolutions, as well as with chunking/batching when not evenly divisible.
# It uses Euclidean distance instead of Manhattan distance, which results in significantly (225%+) longer runtimes in part due to higher memory usage (smaller chunks), but yields a potentially more realistic "round" impact footprint.


import rasterio
import numpy as np
import multiprocessing
import time
import math
import gc
import os
from tempfile import NamedTemporaryFile
import traceback


def load_raster(file_path):
	with rasterio.open(file_path) as dataset:
		return dataset.read(1), dataset.profile, dataset.res

def split_into_batches(lst, batch_count):
	batch_size = len(lst) // batch_count
	remainder = len(lst) % batch_count
	batches = []
	start_index = 0
	for i in range(batch_count):
		# Add an extra chunk to some of the first batches to handle the remainder
		end_index = start_index + batch_size + (1 if i < remainder else 0)
		batches.append(lst[start_index:end_index])
		start_index = end_index
	return batches

def create_lookup_table(max_distance):
	return np.array([1 / (dist**2) if dist > 0 else 0 for dist in range(max_distance)]).astype(np.float32)

# Calculates the distances and impacts of all points in a chunk with NumPy vectorization and the precomputed lookup table
def calculate_impact(args):
	radiant_point_chunk, sample_pixels, lookup_table, radiant_point_raster, resolution_factor = args

	# Calculate Euclidean distances
	diffs = radiant_point_chunk[:, None, :] - sample_pixels[None, :, :]
	radiant_point_distances = np.sqrt((diffs ** 2).sum(axis=2)) * resolution_factor

	# Fit to lookup table indices
	distance_indices = np.round(radiant_point_distances).astype(np.int32)	# int precision dependent on size of lookup table

	# Multiply the impact by the radiant_point intensity
	radiant_point_intensities = radiant_point_raster[radiant_point_chunk[:, 0], radiant_point_chunk[:, 1]]
	partial_impact = np.sum(lookup_table[distance_indices] * radiant_point_intensities[:, None], axis=0)
	# Returns an array representing the radiant impacts from all the radiant_point pixels in the chunk
	return partial_impact

def check_for_stop_file():
	return os.path.exists('stop.txt')



def main():
	start_time = time.time()
	print(f"\nStart time: {time.ctime(start_time)}\n")

	sample_raster_filename = "sample-raster_90m.tif"
	
	# Load initial radiant_point raster info for progress reporting
	initial_radiant_point_raster_filename = "radiance-raster_45m.tif"
	initial_radiant_point_raster_info = load_raster(initial_radiant_point_raster_filename)
	initial_radiant_point_pixel_count = len(np.argwhere(initial_radiant_point_raster_info[0] >= 1))

	# Check for existing radiance-raster (if we should resume a simulation)
	radiant_point_raster_filename = "remaining-radiant_points-raster.tif" if os.path.exists("remaining-radiant_points-raster.tif") else initial_radiant_point_raster_filename

	radiant_point_raster, radiant_point_profile, radiant_point_res = load_raster(radiant_point_raster_filename)
	sample_raster, sample_profile, sample_res = load_raster(sample_raster_filename)
	
	# Scale sample raster for low-res
	scaling_factor = sample_res[0] / radiant_point_res[0]

	# Raster coords are scaled based on the ranges they were created with. Possible to make this more robust in the future?
	sample_pixels = (np.argwhere(sample_raster == 1) * scaling_factor).astype(np.int32)
	radiant_point_pixels = np.argwhere(radiant_point_raster >= 1).astype(np.int32)
	sample_pixel_count = len(sample_pixels)
	radiant_point_pixel_count = len(radiant_point_pixels)
	

	# Initial Report
	print(f"radiant_point Raster dimensions:\t\t{radiant_point_raster.shape[1]}x{radiant_point_raster.shape[0]}\t{round((radiant_point_raster.shape[1] * radiant_point_raster.shape[0]) / 1000000, 1)}MP")
	print(f"Sample Raster dimensions:\t{sample_raster.shape[1]}x{sample_raster.shape[0]}\t{round((sample_raster.shape[1] * sample_raster.shape[0]) / 1000000, 1)}MP")
	print()
	print(f"Sample pixels:\t{sample_pixel_count}")
	print()
	print(f"Initial:")
	print(f"radiant_point pixels\t{initial_radiant_point_pixel_count}")
	print(f"Pixel Pairs\t{sample_pixel_count * initial_radiant_point_pixel_count}")
	print()
	print(f"Remaining ({round((radiant_point_pixel_count / initial_radiant_point_pixel_count) * 100, 2)}%):")
	print(f"radiant_point pixels:\t{radiant_point_pixel_count}")
	print(f"Pixel Pairs:\t{sample_pixel_count * radiant_point_pixel_count}")
	print()


	# (Optional) Select a random subset of radiant_point pixels (0.33 = 1/3rd etc.) (Emitter Resolution)
	proportion = 1
	subset_size = int(len(radiant_point_pixels) * proportion)
	print(f"Selecting {subset_size} random radiant_point pixels ({proportion * 100}%)\n")
	np.random.shuffle(radiant_point_pixels)
	radiant_point_subset = radiant_point_pixels[:subset_size]
	# print(f"Total Pixel Pairs: {subset_size * sample_pixel_count}")


	# Divide pixels into chunks
	# TO DO: Make this robust, i.e. based on raster resolution + available RAM & CPU.
	radiant_point_chunks = np.array_split(radiant_point_subset, (radiant_point_pixel_count / 8))	# This value affects the number of radiant_points per chunk (and memory required)
	chunk_count = len(radiant_point_chunks)

	# Divide chunks into batches; the number of batches should be low enough to avoid significant overhead costs but high enough that memory remains stable and progress is efficient
	total_pairings = radiant_point_pixel_count * sample_pixel_count
	batch_count = math.ceil(chunk_count / 53)		# Raising this boosts CPU usage, but also means memory has more time to add up. Still limited by how much memory can be allocated when results are summed.
	chunk_batches = split_into_batches(radiant_point_chunks, batch_count)
	print(f"Total chunks: {chunk_count}")
	print(f"Total batches: {batch_count}")
	print()
	print(f"radiant_points per chunk: {math.floor(subset_size / chunk_count)}")
	print(f"Chunks per batch: {round(chunk_count / batch_count)}")
	

	# The max possible Euclidean distance is the diagonal distance of the raster
	resolution_factor = 21
	max_distance = np.ceil(np.sqrt(radiant_point_raster.shape[1]**2 + radiant_point_raster.shape[0]**2)).astype(int) * resolution_factor
	lookup_table = create_lookup_table(max_distance)
	

	num_processes = 20
	early_stop = False
	save_point = 0
	processed_pixels = set()
	temp_files = []
	
	print()
	print("Starting impact calculations...")

	try:
		for i, batch in enumerate(chunk_batches):
			with multiprocessing.Pool(processes=num_processes) as pool:
				batch_results = pool.map(calculate_impact, [(chunk, sample_pixels, lookup_table, radiant_point_raster, resolution_factor) for chunk in batch])
			
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
			print(f"Processed {i + 1}/{len(chunk_batches)} batches - Elapsed Time: {elapsed_time:.2f}s, Estimated Time Remaining: {remaining_time:.2f}s")

			# Collects pixels which have been processed
			for chunk in batch:
				processed_pixels.update(map(tuple, chunk))

			# Check for the stop file
			if check_for_stop_file():
				print(f"\nEarly stopping initiated. {round((1 - ((radiant_point_pixel_count - len(processed_pixels)) / initial_radiant_point_pixel_count)) * 100, 2)}% Complete. Saving progress...")
				early_stop = True
				save_point = i
				break


		# Save remaining radiant_point pixels (if any)
		if early_stop:
			# Gather remaining radiant_point pixels
			remaining_radiant_point_pixels = [pixel for pixel in radiant_point_pixels if tuple(pixel) not in processed_pixels]

			# Create a new radiant_point raster
			remaining_radiant_point_raster = np.zeros_like(radiant_point_raster, dtype=radiant_point_raster.dtype)

			# Populate new raster with the remaining radiant_point pixels
			for x, y in remaining_radiant_point_pixels:
				remaining_radiant_point_raster[x, y] = radiant_point_raster[x, y]

			# Save the new radiant_point raster for future processing
			remaining_radiant_point_raster_filename = "remaining-radiant_points-raster.tif"
			with rasterio.open(remaining_radiant_point_raster_filename, 'w', **radiant_point_profile) as dst:
				dst.write(remaining_radiant_point_raster, 1)
			print(f"Remaining radiant_point raster saved as {remaining_radiant_point_raster_filename}")


		# Combine all partial results
		total_impact = sum(np.load(f) for f in temp_files)

		# Combine with previously saved raster data, if it exists
		previous_raster_path = "partial-radiance-raster.tif"
		if os.path.exists(previous_raster_path):
			with rasterio.open(previous_raster_path) as previous_raster:
				# Read and reshape the raster data
				previous_data = previous_raster.read(1).reshape(-1)

			# Indices where previous_data is greater than 0
			indices = np.argwhere(previous_data > 0).flatten()
			total_impact += previous_data[indices]

		# Rescale sample pixels if necessary
		sample_pixels = (sample_pixels / scaling_factor).astype(np.int16)

		# Initialize a new array with the same shape as sample_raster, populate with radiance vals
		total_impact_reshaped = np.zeros_like(sample_raster, dtype=np.float32)
		for (x, y), impact in zip(sample_pixels, total_impact):
			total_impact_reshaped[x, y] = impact

		# Save radiance raster
		save_filename = "partial-radiance-raster.tif" if early_stop else f"final-radiance-raster_{sample_res[0]}m.tif"
		sample_profile['dtype'] = 'float32'
		with rasterio.open(save_filename, 'w', **sample_profile) as dst:
			dst.write(total_impact_reshaped, 1)
		print("Processing complete. Radiance raster created.")
		
	except Exception as e:
		print(f"An error occurred: {e}")
		traceback.print_exc()

	finally:
		# Clean up temporary files
		for f in temp_files:
			os.remove(f)

	elapsed_time = time.time() - start_time
	print(f"\nTotal elapsed time: {elapsed_time:.2f}s\n")


if __name__ == "__main__":
	main()

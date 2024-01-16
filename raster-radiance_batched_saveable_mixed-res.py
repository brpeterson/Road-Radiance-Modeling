# This script is meant to utilize optimizations like parallelization, Manhattan distance, a radiant impact lookup table, and random sparsity in order to approximate the radiant impacts of roads efficiently.
# This script also uses batching to incrementally save to disk, preventing memory from building up too much over time
# This version of the script also allows for stopping / resuming in order to free computer resources in the middle of lengthy simulations, handling of alternate-resolution sample rasters, as well as more detailed reports


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
	return [lst[i * batch_size:(i + 1) * batch_size] for i in range(batch_count)]

def create_lookup_table(max_distance):
	return np.array([1 / (dist**2) if dist > 0 else 0 for dist in range(max_distance)]).astype(np.float32)

# Calculates the distances and impacts of all points in a chunk with NumPy vectorization and the precomputed lookup table
def calculate_impact(args):
	road_chunk, sample_pixels, lookup_table, road_raster = args

	road_distances = np.abs(road_chunk[:, None, :] - sample_pixels[None, :, :]).sum(axis=2)

	# Multiply the impact by the road intensity
	road_intensities = road_raster[road_chunk[:, 0], road_chunk[:, 1]]
	partial_impact = np.sum(lookup_table[road_distances] * road_intensities[:, None], axis=0)
	# Returns an array representing the radiant impacts from all the road pixels in the chunk
	return partial_impact

def check_for_stop_file():
	return os.path.exists('stop.txt')



def main():
	start_time = time.time()
	print(f"\nStart time: {time.ctime(start_time)}\n")

	sample_raster_filename = "Res-180m_Sample-Raster.tif"
	
	# Load initial road raster info for progress reporting
	initial_road_raster_filename = "Res-45m_Road-Raster.tif"
	initial_road_raster_info = load_raster(initial_road_raster_filename)
	initial_road_pixel_count = len(np.argwhere(initial_road_raster_info[0] >= 1))

	# Check for existing radiance-raster (if we should resume a simulation)
	road_raster_filename = "remaining-roads-raster.tif" if os.path.exists("remaining-roads-raster.tif") else initial_road_raster_filename

	road_raster, road_profile, road_res = load_raster(road_raster_filename)
	sample_raster, sample_profile, sample_res = load_raster(sample_raster_filename)

	# For scaling the sample raster, in case of alternative resolutions
	scaling_factor = road_res[0] / sample_res[0]	# Assumes square resolution

	# Raster coords are typed based on the raster sizes they were created with: in this case, int16 will accommodate this resolution. 
	# Has a positive impact on memory usage. Possible to make this more robust in the future?
	sample_pixels = (np.argwhere(sample_raster == 1) * scaling_factor).astype(np.int16)
	road_pixels = np.argwhere(road_raster >= 1).astype(np.int16)
	sample_pixel_count = len(sample_pixels)
	road_pixel_count = len(road_pixels)
	

	# Initial Report
	print(f"Road Raster dimensions:\t\t{road_raster.shape[1]}x{road_raster.shape[0]}\t{round((road_raster.shape[1] * road_raster.shape[0]) / 1000000, 1)}MP")
	print(f"Sample Raster dimensions:\t{sample_raster.shape[1]}x{sample_raster.shape[0]}\t{round((road_raster.shape[1] * road_raster.shape[0]) / 1000000, 1)}MP")
	print()
	print(f"Sample pixels:\t{sample_pixel_count}")
	print()
	print(f"Initial:")
	print(f"Road pixels\t{initial_road_pixel_count}")
	print(f"Pixel Pairs\t{sample_pixel_count * initial_road_pixel_count}")
	print()
	print(f"Remaining ({round((road_pixel_count / initial_road_pixel_count) * 100, 2)}%):")
	print(f"Road pixels:\t{road_pixel_count}")
	print(f"Pixel Pairs:\t{sample_pixel_count * road_pixel_count}")
	print()


	# (Optional) Select a random subset of road pixels (0.33 = 1/3rd etc.) (Emitter Resolution)
	proportion = 1
	subset_size = int(len(road_pixels) * proportion)
	print(f"Selecting {subset_size} random road pixels ({proportion * 100}%)\n")
	np.random.shuffle(road_pixels)
	road_subset = road_pixels[:subset_size]
	# print(f"Total Pixel Pairs: {subset_size * sample_pixel_count}")


	# Divide pixels into chunks
	# TO DO: Make this robust, i.e. based on raster resolution + available RAM & CPU.
	road_chunks = np.array_split(road_subset, (road_pixel_count / 19))	# This value affects the number of roads per chunk (and memory required)
	chunk_count = len(road_chunks)

	# Divide chunks into batches; the number of batches should be low enough to avoid significant overhead costs but high enough that memory remains stable and progress is efficent
	total_pairings = road_pixel_count * sample_pixel_count
	batch_count = math.ceil(chunk_count / 160)		# Raising this boosts CPU usage, but also means memory has more time to add up. Still limited by how much memory can be allocated when results are summed.
	chunk_batches = split_into_batches(road_chunks, batch_count)
	print(f"Total chunks: {chunk_count}")
	print(f"Total batches: {batch_count}")
	print()
	print(f"Roads per chunk: {math.floor(subset_size / chunk_count)}")
	print(f"Chunks per batch: {round(chunk_count / batch_count)}")
	
	
	# # Investigate how much memory chunks are using
	# print("Printing chunk sizes: ")
	# for i in range(len(road_chunks)):
	# 	if i % (round(len(road_chunks))/10) == 0:
	# 		print(f"i: {i}")
	# 		print(f"Chunk {i} - {road_chunks[i].nbytes}")
	# input()

	# The max Manhattan distance is the length + width of the raster
	max_distance = road_raster.shape[1] + road_raster.shape[0]
	lookup_table = create_lookup_table(max_distance)


	num_processes = 20
	early_stop = False
	save_point = 0
	
	print()
	print("Starting impact calculations...")
	processed_pixels = set()
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

			# Collects pixels which have been processed
			for chunk in batch:
				processed_pixels.update(map(tuple, chunk))

			# Check for the stop file
			if check_for_stop_file():
				# print(f"{round((1 - (road_pixel_count / initial_road_pixel_count)) * 100, 2)}% Complete.\n")
				print(f"\nEarly stopping initiated. {round((1 - ((road_pixel_count - len(processed_pixels)) / initial_road_pixel_count)) * 100, 2)}% Complete. Saving progress...")
				early_stop = True
				save_point = i
				break


		# Save remaining road pixels (if any)
		if early_stop:
			# Gather remaining road pixels
			remaining_road_pixels = [pixel for pixel in road_pixels if tuple(pixel) not in processed_pixels]

			# Create a new road raster
			remaining_road_raster = np.zeros_like(road_raster, dtype=road_raster.dtype)

			# Populate new raster with the remaining road pixels
			for x, y in remaining_road_pixels:
				remaining_road_raster[x, y] = road_raster[x, y]

			# Save the new road raster for future processing
			remaining_road_raster_filename = "remaining-roads-raster.tif"
			with rasterio.open(remaining_road_raster_filename, 'w', **road_profile) as dst:
				dst.write(remaining_road_raster, 1)
			print(f"Remaining road raster saved as {remaining_road_raster_filename}")


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
		save_filename = "partial-radiance-raster.tif" if early_stop else "final-radiance-raster.tif"
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

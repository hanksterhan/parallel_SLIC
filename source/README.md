This is the top-level directory for your project source code

Use nvidia-smi to watch the GPU

## workflows
 - boundary recall metric
 - timing
 - debug: why don't big(ish) images work?
 - debug: why doesn't our code produce slic-like segments
 - debug: todos
 - slico
 - enforce_connectivity on cuda_labels
 - abstract
 - other implementations of kernals (ex. per pixel recompute_centroids)
 - use pre-processing kernal
 - refactor code to make it easier to time different implementations
 - display superpixels as average values

## To use boundary recall:
 - Choose image from the Extended Berkeley Segmentation Benchmark and save it in boundary_recall/original_images
 - Run SLIC.py with the desired command line arguments plus the following:
 - Include -b flag
 - Include -f flag followed by boundary_recall/boundaries plus original image name plus .png extension
 - Save ground truth image as as matlab file in boundary_recall/ground_truth
 - Run MatLab and navigate to the folder boundary_recall/benchmarks and run boundary_recall.m
 - The third column of boundary_recall/results/eval_bdry_img.txt includes the boundary recall of each image based on the ground truth
 - The sixth column of boundary_recall/results/eval_bdry.txt is the per image, best value of the measure over all available ground truth segmentations used and averaged over all images. 

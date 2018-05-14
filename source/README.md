This is the top-level directory for your project source code

Use nvidia-smi to watch the GPU

## workflows
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
 - Choose image from the Extended Berkeley Segmentation Benchmark and save it in boundary_recall/original_images (ex. boundary_recall/original_images/3063.jpg)
 - Run SLIC.py with the desired command line arguments plus the following:
   - Include -b flag
   - Include -f flag followed by boundary_recall/boundaries plus original image name plus .png extension (ex. boundary_recall/boundaries/3063.png)
 - Save ground truth image as a matlab file in boundary_recall/ground_truth with the original image name (ex. boundary_recall/ground_truth/3063.mat)
 - Run MatLab and navigate to the folder boundary_recall/benchmarks and run boundary_recall.m
 - The third column of boundary_recall/results/eval_bdry_img.txt includes the boundary recall of each image based on the ground truth
 - The sixth column of boundary_recall/results/eval_bdry.txt is the per image, best value of the measure over all available ground truth segmentations used and averaged over all images. 
 
 Note: The 3 file must all have the same name, just with different file extensions (ex. 3063.jpg, 3063.png, 3063.mat)
 \nNote: boundary_recall.m can run benchmarks on multiple pictures at the same time, appropriate naming is crucial to make this work properly. 

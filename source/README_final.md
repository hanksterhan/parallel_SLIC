## To run SLIC Superpixel Algorithm
 
    $python SLIC.py -k 100 -i input/flower.jpg

To run our parallel implementation of SLIC, include the -p flag:
    $python SLIC.py -k 100 -i input/flower.jpg -p
 
k is the number of superpixels and can be adjusted. The filepath of the image to be segmented should follow the -i flag. For a more detailed breakdown of command line arguments, run $python SLIC.py -h

## To calculate boundary recall:
 - Choose image from the [Extended Berkeley Segmentation Benchmark](https://github.com/davidstutz/extended-berkeley-segmentation-benchmark) and save it in boundary_recall/original_images (ex. boundary_recall/original_images/3063.jpg)
 - Run SLIC.py with the desired command line arguments plus the following:
   - Include -b flag
   - Include -f flag followed by boundary_recall/boundaries plus original image name plus .png extension (ex. boundary_recall/boundaries/3063.png)
    $python SLIC.py -k 100 -i boundary_recall/original_images/3063.jpg -b -f boundary_recall/boundaries/3063.png
 - Save ground truth image as a matlab file in boundary_recall/ground_truth with the original image name (ex. boundary_recall/ground_truth/3063.mat) Note: This is found from the [Extended Berkeley Segmentation Benchmark](https://github.com/davidstutz/extended-berkeley-segmentation-benchmark) as well
 - Run MatLab ("matlab -nodisplay" in terminal to stay in a command line interface) and navigate to the folder boundary_recall/benchmarks and run $boundary_recall.m
 - The third column of boundary_recall/results/eval_bdry_img.txt includes the boundary recall of each image based on the ground truth
 - The sixth column of boundary_recall/results/eval_bdry.txt is the per image, best value of the measure over all available ground truth segmentations used and averaged over all images. 
 
 Note: The 3 files must all have the same name, just with different file extensions (ex. 3063.jpg, 3063.png, 3063.mat)
 
 Note: boundary_recall.m can run benchmarks on multiple pictures at the same time, appropriate naming is crucial to make this work properly. 

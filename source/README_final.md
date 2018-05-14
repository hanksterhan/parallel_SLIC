## Setup

Our project does not require any special environment setup, but the library imports need to execute. This means that libraries such as `PyCUDA` and `numpy` must be installed. This code should be run on a computer that has a GPU.

## To run SLIC
To run the sequential version of SLIC by skimage:

    $ python SLIC.py -k 100 -i input/small.jpg

To run our parallel implementation of SLIC, include the -p flag:

    $ python SLIC.py -k 100 -i input/small.jpg -p

To run SLIC accounting for the compactness factor, m, include the -m flag followed by an integer [1,20]:

    $ python SLIC.py -k 100 -i input/small.jpg -p -m 20

k is the number of superpixels and can be adjusted. The filepath of the image to be segmented should follow the -i flag. Both k and i are mandatory arguments. For a more detailed breakdown of command line arguments, run:

    $ python SLIC.py -h

## To Calculate Boundary Recall:
 - Choose image from the [Extended Berkeley Segmentation Benchmark](https://github.com/davidstutz/extended-berkeley-segmentation-benchmark) and save it in boundary_recall/original_images (ex. boundary_recall/original_images/3063.jpg)
 - Run SLIC.py with the desired command line arguments plus the following:
   - Include -b flag
   - Include -f flag followed by boundary_recall/boundaries plus original image name plus .png extension (ex. boundary_recall/boundaries/3063.png)

         $ python SLIC.py -k 100 -i boundary_recall/original_images/3063.jpg -b -f boundary_recall/boundaries/3063.png

 - Save ground truth image as a matlab file in boundary_recall/ground_truth with the original image name (ex. boundary_recall/ground_truth/3063.mat) Note: This is found from the [Extended Berkeley Segmentation Benchmark](https://github.com/davidstutz/extended-berkeley-segmentation-benchmark) as well
 - Run MatLab ("matlab -nodisplay" in terminal to stay in a command line interface) and navigate to the folder boundary_recall/benchmarks and run:

       $ boundary_recall

 - The third column of boundary_recall/results/eval_bdry_img.txt includes the boundary recall of each image based on the ground truth
 - The sixth column of boundary_recall/results/eval_bdry.txt is the per image, best value of the measure over all available ground truth segmentations used and averaged over all images.

 Note: The 3 files must all have the same name, just with different file extensions (ex. 3063.jpg, 3063.png, 3063.mat)

 Note: boundary_recall.m can run benchmarks on multiple pictures at the same time, appropriate naming is crucial to make this work properly.

 ## Code Structure

  - `anchataEtAlCode/` - contains several of the most relevant files from the C++ SLIC implementation by Anchanta et. al. which we downloaded from a zip file. It is long and not very polished (though we added some comments) but was useful in helping us understand the SLIC algorithm. The main function we referenced is `PerformSuperpixelSLIC` beginning on line 505 in `SLIC.cpp`. It is not possible to run this code as it has dependencies on files not in this directory.
  - `boundary_recall/`
    - `benchmarks/` and `source/`- matlab code for calculating boundary recall, copied from [Extended Berkeley Segmentation Benchmark](https://github.com/davidstutz/extended-berkeley-segmentation-benchmark) git repo
    - `results*/` - txt files with experimental results from different trials
    - `boundaries*/` and `original_images/` - input files for boundary recall experiments
  - `experiments/` - scripts we used to generate timing and boundary recall results as well as resulting files from timing tests. `time04.csv` contains the timing data used in our final report.
  - `input/` - input images
    - `frog*.jpg` - images used to generate timing results
    - `tiny.jpg`, `mini.jpg`, `small.jpg` - main images that we used for debugging
    - `ahuizotl.jpg`, `flower.jpg`, `mountain.jpg`, `obama.jpg` - larger images that we used for debugging
  - `skimageCode/`
    - `cudaSLIC.py` - CUDA kernels used by `slic.py`
    - `rgb2lab.py` - code to do parallelized image pre-processing. Not included in our slic pipeline.
    - `slic.py` - main file containing code for slic. Defines 3 functions:
      - `slic()` - performs image pre and post processing and calls skimage code for sequential segmentation or `slic_cuda` for parallelized segmentation
      - `slic_cuda()` - performs parallelized segmentation by calling kernels in `cudaSLIC.py`
      - `mark_cuda_labels()` - colors image in parallel based on average colors of each superpixel
  - `pycuda_tutorial.py` - test code for learning pycuda, somewhat modified from the original tutorial.
  - `SLIC.py` - RUN THIS FILE. It processes command line arguments, calls `slic()`, and displays the results.

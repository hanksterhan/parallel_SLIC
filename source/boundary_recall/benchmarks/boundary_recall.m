% Test boundary recall for results stored as a soft/hard boundary map:
%
% INPUT
%   imgDir: folder containing original images
%   gtDir:  folder containing ground truth data.
%   pbDir:  folder containing boundary detection results for all the images in imgDir.
%           Format can be one of the following:
%             - a soft or hard boundary map in PNG format.
%             - a collection of segmentations in a cell 'segs' stored in a mat file
%             - an ultrametric contour map in 'doubleSize' format, 'ucm2' stored in a mat file with values in [0 1].
%   outDir: folder where evaluation results will be stored
%       nthresh : Number of points in precision/recall curve.


imgDir = 'original_images/';
gtDir = 'ground_truth/';
pbDir = 'boundaries/';
outDir = 'results/';
mkdir(outDir);
nthresh = 5;

tic;
boundaryBench(imgDir, gtDir, pbDir, outDir, nthresh);
toc;

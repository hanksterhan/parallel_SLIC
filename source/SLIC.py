# import the necessary packages
from skimageCode.slic import slic, mark_cuda_labels
from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import argparse

def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", required = True, help = "Number of superpixels ")
    ap.add_argument("-i", "--img", required = True, help = "Path to the image")
    ap.add_argument("-p", action = "store_true", help = "Run parallel CUDA version")
    ap.add_argument("-o", action = "store_true", help = "Run SLICO (ignores m)")
    ap.add_argument("-c", action = "store_false", help = "Don't enforce connectivity")
    ap.add_argument("-m", "--compactness", default = 10.0, help = "Compactness")
    ap.add_argument("-n", "--iter", default = 10, help = "Number of iterations")
    args = vars(ap.parse_args())

    # load the image and convert it to a floating point data type
    image = img_as_float(io.imread(args["img"]))

    # show initial image
    fig = plt.figure("original image")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image)
    plt.axis("off")

    # RUN SLIC
    print "\nrunning SLIC on %s with k=%s" % (args["img"],  args["k"])
    print "  parallel=%s, compactness=%s, slic_zero=%s" % \
        (args["p"], args["compactness"], args["o"])
    print "  enforce_connectivity=%s, iter=%s\n" % (args["c"], args["iter"])

    # default parameters for slic():
    #   n_segments=100, compactness=10.0, max_iter=10, sigma=0, spacing=None,
    #   multichannel=True, convert2lab=None, enforce_connectivity=True,
    #   min_size_factor=0.5, max_size_factor=3, slic_zero=False
    segments, centroids_dim = slic(
        image,
        n_segments = int(args["k"]),
        parallel = args["p"],
        slic_zero = args["o"],
        enforce_connectivity = args["c"],
        compactness = float(args["compactness"]),
        max_iter = int(args["iter"])
    )

    # display resulting image
    if args["p"]:
        # color image by superpixel averages
        image_cuda = image[np.newaxis, ...]
        image_colored = mark_cuda_labels(image_cuda, centroids_dim, segments)[0]

        # superimpose superpixels onto image
        image_segmented = mark_boundaries(image, segments, mode='inner')

    else:
        # color image by superpixel averages #TODO: make this command line opt
        image_colored = label2rgb(segments, image, kind = "avg")

        # superimpose superpixels onto image
        image_segmented = mark_boundaries(image, segments, mode='inner')

    # show the output of SLIC
    fig = plt.figure("mosaic")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image_segmented)
    plt.axis("off")
    fig = plt.figure("dyed")
    ax2 = fig.add_subplot(1, 1, 1)
    ax2.imshow(image_colored)
    plt.axis("off")

    # show the plots
    plt.show()

if __name__=="__main__":
    main()

# imprt the necessary packages
from skimageCode.slic import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse

def main():
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-i", "--image", required = True, help = "Path to the image")
  args = vars(ap.parse_args())

  # load the image and convert it to a floating point data type
  image = img_as_float(io.imread(args["image"]))

  # loop over different k values (k is the number of superpixels)
  for numSegments in [100]:
    print "running SLIC with k =", numSegments

    # RUN SLIC
    # default parameters for slic():
    #   n_segments=100, compactness=10.0, max_iter=10, sigma=0, spacing=None,
    #   multichannel=True, convert2lab=None, enforce_connectivity=True,
    #   min_size_factor=0.5, max_size_factor=3, slic_zero=False
    segments = slic(image, n_segments=numSegments, sigma=0, compactness=24)

    # superimpose segments onto image
    image_segmented = mark_boundaries(image, segments, mode='outer')

    # show the output of SLIC
    fig = plt.figure("Superpixels -- %d segments" % (numSegments))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image_segmented)
    plt.axis("off")

  # show the plots
  plt.show()

if __name__=="__main__":
  main()

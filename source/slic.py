# import the necessary packages
from skimage.segmentation import slic
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
 
  # loop over the number of segments
  for numSegments in [100, 300, 1000, 3000]:
    # apply SLIC and extract (approximately) the supplied number
    # of segments
    print numSegments
    segments = slic(image, n_segments=numSegments, sigma=0, compactness=20)
    #segments = slic(image, slic_zero=True, n_segments = numSegments)
 
    # show the output of SLIC
    fig = plt.figure("Superpixels -- %d segments" % (numSegments))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, segments, mode='inner'))
    plt.axis("off")
 
  # show the plots
  plt.show()

if __name__=="__main__":
  main()
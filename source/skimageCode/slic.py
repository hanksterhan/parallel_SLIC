# coding=utf-8



import numpy as np
from time import time
from math import ceil

from skimage.util import img_as_float, regular_grid
from skimage.segmentation._slic import (_slic_cython,
                                        _enforce_label_connectivity_cython)
from skimage.color import rgb2lab

from pycuda import driver as cuda
from cudaSLIC import *

# set up printing and logging
np.set_printoptions(threshold = np.inf, linewidth = np.inf)
import logging as lg
lg.basicConfig(level=lg.WARN, format='%(message)s')
#NOTE: set level=lg.DEBUG to see prints, level=lg.WARN to supress prints

"""
slic - performs superpixel segmentation using k-means clustering in
       color-(x,y,z) space.

Parameters:
 - image: 2D, 3D or 4D ndarray, can be grayscale or multichannel
 - parallel: bool, if True run cuda slic, otherwise run skimage serial slic
 - n_segments: int, (approximate) number of superpixels
 - compactness: float, balances color vs space distance. Higher values give
     more weight to space distance, making superpixel shapes more round.
 - max_iter: int, maximum number of iterations of k-means
 - spacing: array-like of 3 floats, controls the weights of the distances
     along z, y, and x dimensions during k-means clustering.
 - multichannel: bool, whether the last axis of the image is to be interpreted
     as multiple channels or another spatial dimension.
 - convert2lab: bool, whether the input should be converted to lab colorspace
     prior to segmentation. The input image *must* be RGB. Highly recommended.
     This option defaults to ``True`` when ``multichannel=True`` *and*
     ``image.shape[-1] == 3``.
 - enforce_connectivity: bool, whether the generated segments must be connected
 - min_size_factor: float, when enforcing connectivity the proportion of the
     minimum segment size to be removed with respect
     to the supposed segment size ```depth*width*height/n_segments```
 - max_size_factor: float, when enforcing connectivity the proportion of the
     maximum connected segment size. A value of 3 works in most of the cases.
 - slic_zero: bool, whether to run SLIC-zero, the zero-parameter mode of SLIC.
     Only possible when not running in parallel.

Returns:
 - labels: array with superpixel index for each pixel
 - centroids_dim: dimensions of the initial grid of centroids

Raises:
 - ValueError: If ``convert2lab`` is set to ``True`` but the last array
    dimension is not of length 3.

Notes:
 - The image is rescaled to be in [0, 1] prior to processing.
 - Images of shape (M, N, 3) are interpreted as 2D RGB images by default. To
  interpret them as 3D with the last dimension having length 3, use
  `multichannel=False`.

References:
 - skimage: https://github.com/scikit-image/scikit-image/blob/v0.13.0/skimage/segmentation/slic_superpixels.py
 - slic: Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi,
    Pascal Fua, and Sabine Süsstrunk, SLIC Superpixels Compared to
    State-of-the-art Superpixel Methods, TPAMI, May 2012.
 - slic-zero: http://ivrg.epfl.ch/research/superpixels#SLICO
"""
def slic(image, parallel=True, n_segments=100, compactness=10., max_iter=10,
         spacing=None, multichannel=True, convert2lab=None,
         enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3,
         slic_zero=False):
    lg.debug("... starting slic.py ...")


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    PRE-PROCESSING
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # reshape image to 3D, record if it was originally 2D
    image = img_as_float(image)
    is_2d = False
    if image.ndim == 2:
        # 2D grayscale image
        image = image[np.newaxis, ..., np.newaxis]
        is_2d = True
    elif image.ndim == 3 and multichannel:
        # Make 2D multichannel image 3D with depth = 1
        image = image[np.newaxis, ...]
        is_2d = True
    elif image.ndim == 3 and not multichannel:
        # Add channel as single last dimension
        image = image[..., np.newaxis]

    # convert RGB -> LAB
    if multichannel and (convert2lab or convert2lab is None):
        if image.shape[-1] != 3 and convert2lab:
            raise ValueError("Lab colorspace conversion requires a RGB image.")
        elif image.shape[-1] == 3:
            image = rgb2lab(image.astype(np.float32))

    # make contiguous is memory
    image = np.ascontiguousarray(image) #zyxc order, float64


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    INITIALIZE PARAMETERS USED FOR SEGMENTATION
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    ########################################################
    # initalize segments, step, and spacing for _slic_cython
    # this section of code comes mostly from the skimage library
    depth, height, width = image.shape[:3]

    # initalize spacing
    if spacing is None:
        spacing = np.ones(3)
    elif isinstance(spacing, (list, tuple)):
        spacing = np.array(spacing, dtype=np.double)

    # initialize cluster centroids for desired number of segments
    grid_z, grid_y, grid_x = np.mgrid[:depth, :height, :width]
    slices = regular_grid(image.shape[:3], n_segments)
    step_z, step_y, step_x = [int(s.step if s.step is not None else 1)
                              for s in slices]
    segments_z = grid_z[slices]
    segments_y = grid_y[slices]
    segments_x = grid_x[slices]

    segments_color = np.zeros(segments_z.shape + (image.shape[3],))
    segments = np.concatenate([segments_z[..., np.newaxis],
                               segments_y[..., np.newaxis],
                               segments_x[..., np.newaxis],
                               segments_color],
                              axis=-1).reshape(-1, 3 + image.shape[3])
    segments = np.ascontiguousarray(segments)

    step = float(max((step_z, step_y, step_x)))

    # ratio is to scale image for _clic_cython, which expects an image
    # that is already scaled by 1/compactness
    ratio = 1.0 / compactness

    ######################################################
    # initalize centroids and centroids_dim for slic_cuda

    # centroids is a 1D array with 6D centroids represented sequentially
    # (example: [l1 a1 b1 x1 y1 z1 l2 a2 b2 x2 y2 z2 l3 a3 b3 x3 y3 z3])
    centroids = np.array([segment[::-1] for segment in segments],
        dtype = np.float32)

    # compute the dimensions of the initial grid of centroids
    centroids_dim = \
        np.array([len(range(slices[n].start, image.shape[n], slices[n].step))
        for n in [2, 1, 0]], dtype=np.int32)


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    SEGMENTATION
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # actual call to slic, with timing
    tstart = time()
    if parallel:
        labels = slic_cuda(image, centroids, centroids_dim, compactness,
            max_iter)
    else:
        labels = _slic_cython(image * ratio, segments, step, max_iter, spacing,
            slic_zero)
        # print "%s, %s, %s," % (0, 0, 0), # for piping into csv
    tend = time()

    if enforce_connectivity:
        #labels = np.ascontiguousarray(labels.astype(np.intp))
        segment_size = depth * height * width / n_segments
        min_size = int(min_size_factor * segment_size)
        max_size = int(max_size_factor * segment_size)
        labels = _enforce_label_connectivity_cython(
            np.ascontiguousarray(labels.astype(np.intp)),
            n_segments,
            min_size,
            max_size
        )

    print "TIME: ", tend-tstart

    if is_2d:
        labels = labels[0]

    return labels, centroids_dim


"""
slic_cuda - performs slic to assign pixels to superpixels given initial
    superpixel centroid locations

Parameters:
 - image: zyxc ordered ndarray of type float64
 - centroids: [k,6] ndarray of type float32, each 6 is ordered labxyz
 - centroids_dim: xyz ordered ndarray of type int32, specifies shape of initial
   centroid grid
 - compactness: int, controls lab vs. xy distance weight
 - max_iter: int, specifies number of iterations

Returns:
 - assignments: zyx ordered ndarray of type int32. Encodes the
   superpixel assignment for each pixel.
"""
def slic_cuda(image, centroids, centroids_dim, compactness, max_iter):
    htod_stream = cuda.Stream()
    kernel_stream = cuda.Stream()

    # initialize structures and copy to GPU
    tstart1 = time()
    image32 = np.ascontiguousarray(np.swapaxes(image, 0, 2).astype(np.float32))
    # image32 is now xyzc order, float32
    img_dim = np.array(image32.shape[:-1], dtype=np.int32)
    # img_dim is now just xyz from image32's xyzc
    centroids_dim_int = centroids_dim.astype(int)
    assignments = np.zeros(image32.shape[:-1], dtype=np.int32)

    image_gpu = cuda.mem_alloc(image32.nbytes)
    img_dim_gpu = cuda.mem_alloc(img_dim.nbytes)
    centroids_gpu = cuda.mem_alloc(centroids.nbytes)
    centroids_dim_gpu = cuda.mem_alloc(centroids_dim.nbytes)
    assignments_gpu = cuda.mem_alloc(assignments.nbytes)

    cuda.memcpy_htod_async(image_gpu, image32, stream=htod_stream)
    cuda.memcpy_htod_async(img_dim_gpu, img_dim, stream=htod_stream)
    cuda.memcpy_htod_async(centroids_gpu, centroids, stream=htod_stream)
    cuda.memcpy_htod_async(centroids_dim_gpu, centroids_dim, stream=htod_stream)
    # don't need to copy assignments to gpu because first_assignments_func
    # initializes this memory

    htod_stream.synchronize() # wait for memcpys to complete
    tend1 = time()
    lg.warn("htod copy time: %s", tend1-tstart1)

    # debug logs
    lg.debug("dims:")
    lg.debug("  img %s", image.shape)
    lg.debug("  img32 %s %s %s %s", image32.shape, img_dim, image32.dtype,
        img_dim.dtype)
    lg.debug("  centroids %s %s %s %s", centroids.shape, centroids_dim,
        centroids.dtype, centroids_dim.dtype)
    lg.debug("  assignments %s %s", assignments.shape, assignments.dtype)

    # done with copies, begin timing and call kernels
    tstart2 = time()

    # initialize pixel-centroid assignments
    first_assignments_func(
        img_dim_gpu,
        centroids_dim_gpu,
        assignments_gpu,
        block=(128,8,1),
        grid=(int(ceil(float(image32.shape[0])/128)),
            int(ceil(float(image32.shape[1])/8)), image32.shape[2]),
        stream=kernel_stream
    )
    # cuda.memcpy_dtoh(assignments, assignments_gpu)
    # print assignments

    # debug logs
    # cuda.memcpy_dtoh(assignments, assignments_gpu)
    # lg.debug("INITAL assignments:")
    # lg.debug(np.swapaxes(assignments, 0, 2))
    # lg.debug("INITIAL centroids:")
    # lg.debug(centroids)

    # iteratively updated centroids and assignments
    for i in range(max_iter):
        recompute_centroids_func(
            image_gpu,
            img_dim_gpu,
            centroids_gpu,
            centroids_dim_gpu,
            assignments_gpu,
            block=(128,8,1),
            grid=(int(ceil(float(centroids_dim_int[0])/128)),
                int(ceil(float(centroids_dim_int[1])/8)), centroids_dim_int[2]),
            stream=kernel_stream
        )

        update_assignments_func(
            image_gpu,
            img_dim_gpu,
            centroids_gpu,
            centroids_dim_gpu,
            assignments_gpu,
            np.int32(compactness),
            block=(128,8,1),
            grid=(int(ceil(float(image32.shape[0])/128)),
                int(ceil(float(image32.shape[1])/8)), image32.shape[2]),
            stream=kernel_stream
        )

    kernel_stream.synchronize() # wait for kernels to complete
    tend2 = time()
    lg.warn("   kernel time: %s", tend2-tstart2)

    tstart3 = time()
    cuda.memcpy_dtoh(assignments, assignments_gpu)
    tend3 = time()
    lg.warn("dtoh copy time: %s\n", tend3-tstart3)

    assignments = np.swapaxes(assignments, 0, 2)
    #print assignments
    lg.debug("FINAL assignments:")
    lg.debug(assignments)

    # for printing to csv
    # print "%s, %s, %s," % (tend1-tstart1, tend2-tstart2, tend3-tstart3),

    return assignments

"""
mark_cuda_labels - superimpose segments onto image

Parameters:
 - image: zyxc ordered ndarray
 - centroids_dim: xyz ordered ndarray of type int32, specifies shape of initial
    centroid grid
 - assignments: yx ordered ndarray

Returns:
 - final_image: zyxc ordered ndarray with pixels set to centroid values
"""
def mark_cuda_labels(image, centroids_dim, assignments):
    lg.debug("mark_cuda_labels on image, dims = %s", image.shape)

    # initialize structures and copy to GPU
    image32 = np.ascontiguousarray(np.swapaxes(image, 0, 2).astype(np.float32))
    # image32 is now in xyzc order, float32
    img_dim = np.array(image32.shape[:-1], dtype=np.int32)
    # img_dim is now just xyz from image32's xyzc
    centroids = np.empty([np.product(centroids_dim),6], dtype=np.float32)
    centroids_dim = centroids_dim.astype(np.int32)
    assignments = assignments.astype(np.int32)
    centroids_dim_int = centroids_dim.astype(int)

    image_gpu = cuda.mem_alloc(image32.nbytes)
    img_dim_gpu = cuda.mem_alloc(img_dim.nbytes)
    centroids_gpu = cuda.mem_alloc(centroids.nbytes)
    centroids_dim_gpu = cuda.mem_alloc(centroids_dim.nbytes)
    assignments_gpu = cuda.mem_alloc(assignments.nbytes)

    cuda.memcpy_htod(image_gpu, image32)
    cuda.memcpy_htod(img_dim_gpu, img_dim)
    #cuda.memcpy_htod(centroids_gpu, centroids)
    cuda.memcpy_htod(centroids_dim_gpu, centroids_dim)
    cuda.memcpy_htod(assignments_gpu, assignments)

    # use recompute_centroids to find average rgb values
    recompute_centroids_func(
        image_gpu,
        img_dim_gpu,
        centroids_gpu,
        centroids_dim_gpu,
        assignments_gpu,
        block=(128,8,1),
        grid=(int(ceil(float(centroids_dim_int[0])/128)),
            int(ceil(float(centroids_dim_int[1])/8)), centroids_dim_int[2])
    )

    # set pixel color based on the computed averages
    average_color_func(
        image_gpu,
        img_dim_gpu,
        centroids_gpu,
        assignments_gpu,
        block=(128,8,1),
        grid=(int(ceil(float(image32.shape[0])/128)),
            int(ceil(float(image32.shape[1])/8)), image32.shape[2])
    )

    final_image = np.empty_like(image32)
    cuda.memcpy_dtoh(final_image, image_gpu)
    final_image = np.swapaxes(final_image, 0, 2).astype(float)

    return final_image

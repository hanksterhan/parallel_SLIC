# coding=utf-8

import collections as coll
import numpy as np
from scipy import ndimage as ndi

from skimage.util import img_as_float, regular_grid
from skimage.color import rgb2lab
from pycuda import driver as cuda
from pycuda.compiler import SourceModule
from pycuda import autoinit

from cudaSLIC import *

import matplotlib.pyplot as plt

import pyximport
pyximport.install()
from cython_slic import (_slic_cython, _enforce_label_connectivity_cython)

# set up printing and logging
np.set_printoptions(threshold = np.inf, linewidth = np.inf)
import logging as lg
lg.basicConfig(level=lg.WARN, format='%(message)s')
#NOTE: set level=lg.DEBUG to see prints, level=lg.WARN to supress prints

def slic(image, parallel=True, n_segments=100, compactness=10., max_iter=10, sigma=0,
         spacing=None, multichannel=True, convert2lab=None,
         enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3,
         slic_zero=False):
    """Segments image using k-means clustering in Color-(x,y,z) space.
    Parameters
    ----------
    image : 2D, 3D or 4D ndarray
        Input image, which can be 2D or 3D, and grayscale or multichannel
        (see `multichannel` parameter).
    parallel : bool, if True run cuda slic, otherwise run skimage serial slic
    n_segments : int, optional
        The (approximate) number of labels in the segmented output image.
    compactness : float, optional
        Balances color proximity and space proximity. Higher values give
        more weight to space proximity, making superpixel shapes more
        square/cubic. In SLICO mode, this is the initial compactness.
        This parameter depends strongly on image contrast and on the
        shapes of objects in the image. We recommend exploring possible
        values on a log scale, e.g., 0.01, 0.1, 1, 10, 100, before
        refining around a chosen value.
    max_iter : int, optional
        Maximum number of iterations of k-means.
    sigma : float or (3,) array-like of floats, optional
        Width of Gaussian smoothing kernel for pre-processing for each
        dimension of the image. The same sigma is applied to each dimension in
        case of a scalar value. Zero means no smoothing.
        Note, that `sigma` is automatically scaled if it is scalar and a
        manual voxel spacing is provided (see Notes section).
    spacing : (3,) array-like of floats, optional
        The voxel spacing along each image dimension. By default, `slic`
        assumes uniform spacing (same voxel resolution along z, y and x).
        This parameter controls the weights of the distances along z, y,
        and x during k-means clustering.
    multichannel : bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension.
    convert2lab : bool, optional
        Whether the input should be converted to Lab colorspace prior to
        segmentation. The input image *must* be RGB. Highly recommended.
        This option defaults to ``True`` when ``multichannel=True`` *and*
        ``image.shape[-1] == 3``.
    enforce_connectivity: bool, optional
        Whether the generated segments are connected or not
    min_size_factor: float, optional
        Proportion of the minimum segment size to be removed with respect
        to the supposed segment size ```depth*width*height/n_segments```
    max_size_factor: float, optional
        Proportion of the maximum connected segment size. A value of 3 works
        in most of the cases.
    slic_zero: bool, optional
        Run SLIC-zero, the zero-parameter mode of SLIC. [2]_
    Returns
    -------
    labels : 2D or 3D array
        Integer mask indicating segment labels.
    Raises
    ------
    ValueError
        If ``convert2lab`` is set to ``True`` but the last array
        dimension is not of length 3.
    Notes
    -----
    * If `sigma > 0`, the image is smoothed using a Gaussian kernel prior to
      segmentation.
    * If `sigma` is scalar and `spacing` is provided, the kernel width is
      divided along each dimension by the spacing. For example, if ``sigma=1``
      and ``spacing=[5, 1, 1]``, the effective `sigma` is ``[0.2, 1, 1]``. This
      ensures sensible smoothing for anisotropic images.
    * The image is rescaled to be in [0, 1] prior to processing.
    * Images of shape (M, N, 3) are interpreted as 2D RGB images by default. To
      interpret them as 3D with the last dimension having length 3, use
      `multichannel=False`.
    References
    ----------
    .. [1] Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi,
        Pascal Fua, and Sabine Süsstrunk, SLIC Superpixels Compared to
        State-of-the-art Superpixel Methods, TPAMI, May 2012.
    .. [2] http://ivrg.epfl.ch/research/superpixels#SLICO
    Examples
    --------
    >>> from skimage.segmentation import slic
    >>> from skimage.data import astronaut
    >>> img = astronaut()
    >>> segments = slic(img, n_segments=100, compactness=10)
    Increasing the compactness parameter yields more square regions:
    >>> segments = slic(img, n_segments=100, compactness=20)
    """
    lg.debug("... starting slic.py ...")

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

    # save image before beginning to transform it so it can be displayed later
    original_image = image

    # initalize spacing
    if spacing is None:
        spacing = np.ones(3)
    elif isinstance(spacing, (list, tuple)):
        spacing = np.array(spacing, dtype=np.double)

    # if sigma is set, perform gaussian smoothing
    if not isinstance(sigma, coll.Iterable):
        sigma = np.array([sigma, sigma, sigma], dtype=np.double)
        sigma /= spacing.astype(np.double)
    elif isinstance(sigma, (list, tuple)):
        sigma = np.array(sigma, dtype=np.double)
    if (sigma > 0).any():
        # add zero smoothing for multichannel dimension
        sigma = list(sigma) + [0]
        image = ndi.gaussian_filter(image, sigma)

    # convert RGB -> LAB
    if multichannel and (convert2lab or convert2lab is None):
        if image.shape[-1] != 3 and convert2lab:
            raise ValueError("Lab colorspace conversion requires a RGB image.")
        elif image.shape[-1] == 3:
            image = rgb2lab(image.astype(np.float32))

    depth, height, width = image.shape[:3]

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

    # we do the scaling of ratio in the same way as in the SLIC paper
    # so the values have the same meaning
    step = float(max((step_z, step_y, step_x)))
    ratio = 1.0 / compactness

    image = np.ascontiguousarray(image) #zyxc order, float64

    centroids = np.array([segment[::-1] for segment in segments], dtype = np.float32)
    #centroids is now a 1D array with 6D centroids represented sequentially
    #(example: [l1 a1 b1 x1 y1 z1 l2 a2 b2 x2 y2 z2 l3 a3 b3 x3 y3 z3])
    centroids_dim = np.array([len(range(slices[n].start, image.shape[n], slices[n].step)) for n in [2, 1, 0]], dtype=np.int32)


    if parallel:
        ## PARALLEL ##
        labels = slic_cuda(image, centroids, centroids_dim, compactness)
        #labels = np.ascontiguousarray(labels)

        # display resulting image
        if is_2d:
            cuda_labeled_img = mark_cuda_labels(original_image, centroids_dim, labels[0])
            fig = plt.figure("cuda_labeled_img")
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(cuda_labeled_img[0])
            plt.axis("off")
    else:
        ## SERIAL ##
        lg.debug("about to call _slic_cython with parameters:")
        lg.debug(" > img shape %s", image.shape)
        lg.debug(" > segments shape %s", segments.shape)
        lg.debug(" > step %d", step)
        lg.debug(" > max iter %d", max_iter)
        lg.debug(" > spacing %s %s", spacing.shape, spacing)
        lg.debug(" > slic zero %s", slic_zero)
        labels = _slic_cython(image, segments, step, max_iter, spacing, slic_zero)

    # TODO: do this for cuda_labels as well, currently get error: expected 'Py_ssize_t' but got 'int'
    # if enforce_connectivity:
    #     segment_size = depth * height * width / n_segments
    #     min_size = int(min_size_factor * segment_size)
    #     max_size = int(max_size_factor * segment_size)
    #     labels = _enforce_label_connectivity_cython(
    #         labels,
    #         n_segments,
    #         min_size,
    #         max_size
    #     )

    if is_2d:
        labels = labels[0]

    return labels

"""
slic_cuda - performs slic to assign pixels to superpixels given initial
    superpixel centroid locations

Parameters:
 - image: zyxc ordered ndarray of type float64
 - centroids: [k,6] ndarray of type float32, each 6 is ordered labxyz
 - centroids_dim: xyz ordered ndarray of type int32, specifies shape of initial
   centroid grid

Returns:
 - assignments: zyx ordered ndarray of type int32
"""
def slic_cuda(image, centroids, centroids_dim, compactness):
    image32 = np.ascontiguousarray(np.swapaxes(image, 0, 2).astype(np.float32)) #xyzc order, float32
    # # try making image white on GPU TODO: remove this test code and remove top import
    # white_func(image_gpu, img_dim_gpu, block=(128,8,1), grid=(image32.shape[0], image32.shape[1], image32.shape[2]))
    # new_image = np.empty_like(image32)
    # print "new_image shape:", new_image.shape
    # cuda.memcpy_dtoh(new_image, image_gpu)
    # fig = plt.figure("white? image")
    # ax = fig.add_subplot(1, 1, 1)
    # ax.imshow(np.swapaxes(new_image, 0, 2)[0])
    # plt.axis("off")

    # copy image information to GPU
    image_gpu = cuda.mem_alloc(image32.nbytes)
    cuda.memcpy_htod(image_gpu, image32)

    # copy image dim information to GPU
    img_dim = np.array(image32.shape[:-1], dtype=np.int32) # indexing to just get xyz from xyzc
    img_dim_gpu = cuda.mem_alloc(img_dim.nbytes)
    cuda.memcpy_htod(img_dim_gpu, img_dim)

    # copy centroids to GPU
    centroids_gpu = cuda.mem_alloc(centroids.nbytes)
    cuda.memcpy_htod(centroids_gpu, centroids)

    # figure out the number of centroids spaced along each axis and copy to GPU
    centroids_dim_int = centroids_dim.astype(int)
    centroids_dim_gpu = cuda.mem_alloc(centroids_dim.nbytes)
    cuda.memcpy_htod(centroids_dim_gpu, centroids_dim)

    # copy (uninitialized) assignments to GPU
    assignments = np.zeros(image32.shape[:-1], dtype=np.int32)
    assignments_gpu = cuda.mem_alloc(assignments.nbytes) # this could also be image32.nbytes / 4 if converting to int is costly
    cuda.memcpy_htod(assignments_gpu, assignments)

    # debug logs
    lg.debug("dims:")
    lg.debug("  img %s", image.shape)
    lg.debug("  img32 %s %s %s %s", image32.shape, img_dim, image32.dtype, img_dim.dtype)
    lg.debug("  centroids %s %s %s %s", centroids.shape, centroids_dim, centroids.dtype, centroids_dim.dtype)
    lg.debug("  assignments %s %s", assignments.shape, assignments.dtype)

    # initialize pixel-centroid assignments
    first_assignments_func(
        img_dim_gpu,
        centroids_dim_gpu,
        assignments_gpu,
        block=(128,8,1),
        grid=(image32.shape[0], image32.shape[1], image32.shape[2])
    )

    # debug logs
    cuda.memcpy_dtoh(assignments, assignments_gpu)
    lg.debug("INITAL assignments:")
    lg.debug(np.swapaxes(assignments, 0, 2))
    lg.debug("INITIAL centroids:")
    lg.debug(centroids)

    # iterate 10 times as this is generally enough for convergence
    for i in range(10):
        recompute_centroids_func(
            image_gpu,
            img_dim_gpu,
            centroids_gpu,
            centroids_dim_gpu,
            assignments_gpu,
            block=(64,8,1),
            grid=(centroids_dim_int[0], centroids_dim_int[1], centroids_dim_int[2])
        )

        update_assignments_func(
            image_gpu,
            img_dim_gpu,
            centroids_gpu,
            centroids_dim_gpu,
            assignments_gpu,
            np.int32(compactness),
            block=(64,8,1),
            grid=(image32.shape[0], image32.shape[1], image32.shape[2])
        )

    cuda.memcpy_dtoh(assignments, assignments_gpu)
    assignments = np.swapaxes(assignments, 0, 2)
    lg.debug("FINAL assignments:")
    lg.debug(assignments)

    return assignments

"""
mark_cuda_labels - superimpose segments onto image

Parameters:
 - image: zyxc ordered ndarray
 - centroids_dim: xyz ordered ndarray of type int32, specifies shape of initial centroid grid
 - assignments: yx ordered ndarray

Returns:
 - final_image: zyxc ordered ndarray with pixels set to centroid values
"""
def mark_cuda_labels(image, centroids_dim, assignments):
    lg.debug("mark_cuda_labels on image, dims = %s", image.shape)

    # initialize structures and copy to GPU
    image32 = np.ascontiguousarray(np.swapaxes(image, 0, 2).astype(np.float32)) #xyzc order, float32
    img_dim = np.array(image32.shape[:-1], dtype=np.int32) # indexing to get xyz from xyzc
    centroids = np.empty([np.product(centroids_dim),6], dtype=np.float32)
    centroids_dim_int = centroids_dim.astype(int)

    image_gpu = cuda.mem_alloc(image32.nbytes)
    img_dim_gpu = cuda.mem_alloc(img_dim.nbytes)
    centroids_gpu = cuda.mem_alloc(centroids.nbytes)
    centroids_dim_gpu = cuda.mem_alloc(centroids_dim.nbytes)
    assignments_gpu = cuda.mem_alloc(assignments.nbytes)

    cuda.memcpy_htod(image_gpu, image32)
    cuda.memcpy_htod(img_dim_gpu, img_dim)
    cuda.memcpy_htod(centroids_gpu, centroids)
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
        grid=(centroids_dim_int[0], centroids_dim_int[1], centroids_dim_int[2])
    )

    # set pixel color based on the computed averages
    average_color_func(
        image_gpu,
        img_dim_gpu,
        centroids_gpu,
        assignments_gpu,
        block=(128,8,1),
        grid=(image32.shape[0], image32.shape[1], image32.shape[2])
    )

    final_image = np.empty_like(image32)
    cuda.memcpy_dtoh(final_image, image_gpu)
    final_image = np.swapaxes(final_image, 0, 2).astype(float)

    return final_image

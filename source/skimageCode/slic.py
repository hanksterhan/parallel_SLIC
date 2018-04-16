# coding=utf-8

import collections as coll
import numpy as np
from scipy import ndimage as ndi

from skimage.util import img_as_float, regular_grid
from skimage.color import rgb2lab
from pycuda import driver as cuda
from pycuda.compiler import SourceModule
from pycuda import autoinit

import pyximport
pyximport.install()
from cython_slic import (_slic_cython, _enforce_label_connectivity_cython)


def slic(image, n_segments=100, compactness=10., max_iter=10, sigma=0,
         spacing=None, multichannel=True, convert2lab=None,
         enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3,
         slic_zero=False):
    """Segments image using k-means clustering in Color-(x,y,z) space.
    Parameters
    ----------
    image : 2D, 3D or 4D ndarray
        Input image, which can be 2D or 3D, and grayscale or multichannel
        (see `multichannel` parameter).
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
    print "... starting slic.py ..."

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

    if spacing is None:
        spacing = np.ones(3)
    elif isinstance(spacing, (list, tuple)):
        spacing = np.array(spacing, dtype=np.double)

    if not isinstance(sigma, coll.Iterable):
        sigma = np.array([sigma, sigma, sigma], dtype=np.double)
        sigma /= spacing.astype(np.double)
    elif isinstance(sigma, (list, tuple)):
        sigma = np.array(sigma, dtype=np.double)
    if (sigma > 0).any():
        # add zero smoothing for multichannel dimension
        sigma = list(sigma) + [0]
        image = ndi.gaussian_filter(image, sigma)

    if multichannel and (convert2lab or convert2lab is None):
        if image.shape[-1] != 3 and convert2lab:
            raise ValueError("Lab colorspace conversion requires a RGB image.")
        elif image.shape[-1] == 3:
            image = rgb2lab(image)

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
    #ratio = 1.0 #TODO: remove

    image = np.ascontiguousarray(image * ratio)

    # copy image to GPU
    image_gpu = cuda.mem_alloc(image.nbytes)
    cuda.memcpy_htod(image_gpu, image)


    # try making image white on GPU TODO: remove this test code
    cuda_white_string = SourceModule(
    """
    //# This code should be run with one thread per pixel (max img size is 4096x4096)
    //# makes whole image white
    __global__ void make_white(float* img) {

        // convert from thread+block indices to 1D image index (idx)
        int bx, by, bz, tx, ty, tz, tidx, bidx, idx;
        bx = blockIdx.x;
        by = blockIdx.y;
        bz = blockIdx.z;
        tx = threadIdx.x;
        ty = threadIdx.y;
        tz = threadIdx.z;
        tidx = tx + ty * blockDim.x + tz * blockDim.x * blockDim.y;
        bidx = bx + by * gridDim.x  + bz * gridDim.x  * gridDim.y;
        idx = tidx + bidx * blockDim.x * blockDim.y * blockDim.z;

        // use idx to set all pixels to white
        img[3 * idx + 0] = (float) tx; // L
        //img[3 * idx + 1] = (float) 0;   // a
        //img[3 * idx + 2] = (float) 0;   // b

    }""")
    white_func = cuda_white_string.get_function("make_white")
    white_func(image_gpu, block=(image.shape[2], image.shape[1], image.shape[0]))
    new_image = np.empty_like(image)
    cuda.memcpy_dtoh(new_image, image_gpu)
    # print "image"
    # print image.astype(np.int)
    # print "new image"
    # print new_image.astype(np.int)
    # pi, pj, pk, _ = image.shape
    # for pii in range(pi):
    #     for pjj in range(pj):
    #         print "---"
    #         for pkk in range(pk):
    #             print image[pii][pjj][pkk].astype(np.int), "   |||   ", new_image[pii][pjj][pkk].astype(np.int)


    # copy initial centroids to GPU
    print "segments shape: ", segments.shape
    centroids = np.array([])
    for segment in segments:
        for s in segment:
            centroids.add(s)
    centroids = centroids.astype(int)
    #centroids is now a 1D array with 6D centroids represented sequentially
    #(example: [l1 a1 b1 x1 y1 z1 l2 a2 b2 x2 y2 z2 l3 a3 b3 x3 y3 z3])
    #TODO: figure out the number of centroids spaced along x and y axes?
    centroids_gpu = cuda.mem_alloc(centroids.nbytes)
    cuda.memcpy_htod(centroids_gpu, centroids)

    print "about to call _slic_cython with parameters:"
    print " > img shape", image.shape
    print " > segments shape", segments.shape
    print " > step", step
    print " > max iter", max_iter
    print " > spacing", spacing.shape, spacing
    print " > slic zero", slic_zero
    labels = _slic_cython(image, segments, step, max_iter, spacing, slic_zero)
    # Params: image, alloc'd on GPU: float*
    #         image_x, image_y, image_z, xyz dimensions of image: int
    #         segments, alloc'd on GPU: int*

    #labels = cuda_slic_cython(image, image.shape)

    if enforce_connectivity:
        segment_size = depth * height * width / n_segments
        min_size = int(min_size_factor * segment_size)
        max_size = int(max_size_factor * segment_size)
        labels = _enforce_label_connectivity_cython(labels,
                                                    n_segments,
                                                    min_size,
                                                    max_size)

    if is_2d:
        labels = labels[0]

    return labels

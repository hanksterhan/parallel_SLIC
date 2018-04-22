from pycuda.compiler import SourceModule

white_func = SourceModule(
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
      img[3 * idx + 0] = (float) 1; // R
      img[3 * idx + 1] = (float) 1; // G
      img[3 * idx + 2] = (float) 0; // B

  }""").get_function("make_white")

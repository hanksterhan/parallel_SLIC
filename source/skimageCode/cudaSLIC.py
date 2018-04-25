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

update_assignments_func = SourceModule(
  """
  //# This code should be run with one thread per pixel (max img size is 4096x4096)
  //# Responsible to updating pixel to superpixel assignments based on new centroids
  __global__ void update_assignments(float* img, int* img_dim, float* cents, int* cents_dim, int* assignment) {
    int x, y, z, n;
    x = img_dim[0];
    y = img_dim[1];
    z = img_dim[2];
    n = x * y * z;

    //# get 1D pixel index from thread+block indices
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

    //# don't try to act if your id is out of bounds of the picture
    if(idx >= n){
        return;
    }

    //# get pixel 3D indices from 1D idx and img_dim
    int px, py, pz;
    px = idx % x;
    py = idx / x;
    pz = idx / (x * y);

    //# get pixel lab values
    int pl, pa, pb;
    pl = img[3 * idx + 0];
    pa = img[3 * idx + 1];
    pb = img[3 * idx + 2];

    //# get centroid 1D (cent) and 3D (cx, cy, cz) indices
    int cent, cx, cy, cz;
    cent = assignment[idx];
    cx = cent % cents_dim[0];
    cy = cent / cents_dim[0];
    cz = cent / (cents_dim[0] * cents_dim[1]);

    //# loop over (up to) 27 nearby centroids and reassign if necessary
    int f, g, h, kidx, kl, ka, kb, kx, ky, kz;
    double dist, dist_lab, dist_xyz, min_dist;
    min_dist = 999999; //#TODO: maybe make maxfloat
    //# maybe: CUDART_INF_F or CUDART_INF defined in /usr/local/cuda/include/math_constants.h

    for(f = cx-1; f <= cx+1; f++){
        for(g = cy-1; g <= cy+1; g++){
            for(h = cz-1; h <= cz+1; h++){
                //# check bounds
                if(f>=0 && g>=0 && h>=0 && f<=cents_dim[0] && g<=cents_dim[1] && h<=cents_dim[2]){
                    //# get centroid 1D indices from f, g, h, and cents_dim
                    kidx = f + g * cents_dim[0] + h * cents_dim[0] * cents_dim[1];
                    kl = cents[6 * kidx + 0];
                    ka = cents[6 * kidx + 1];
                    kb = cents[6 * kidx + 2];
                    kx = cents[6 * kidx + 3];
                    ky = cents[6 * kidx + 4];
                    kz = cents[6 * kidx + 5];

                    //# for each compute dist
                    dist_lab = (pl - kl)*(pl - kl) + (pa - ka)*(pa - ka) + (pb - kb)*(pb - kb);
                    dist_xyz = (px - kx)*(px - kx) + (py - ky)*(py - ky) + (pz - kz)*(pz - kz);
                    dist = dist_lab + dist_xyz; //#this is approximate, doesnt include sqrts

                    //# if this centroid is closer, update assignment
                    if (dist < min_dist){
                        assignment[idx] = kidx;
                        min_dist = dist;
                    }
                }
            }
        }
    }
  }""").get_function("update_assignments")

first_assignment_func = SourceModule("""
__global__ void first_assignments(int* img_dim, int* cents_dim, int* assignments){
    //# get image dimensions
    int x, y, z, n;
    x = img_dim[0];
    y = img_dim[1];
    z = img_dim[2];
    n = x * y * z;

    //# get centroid seeds dimensions
    int cx, cy, cz;
    cx = cents_dim[0];
    cy = cents_dim[1];
    cz = cents_dim[2];

    //# get 1D pixel index from thread+block indices
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

    //# don't try to act if your id is out of bounds of the picture
    if(idx >= n){
        return;
    }

    //# get pixel 3D indices from 1D idx and img_dim
    int px, py, pz;
    px = idx % x;
    py = idx / x;
    pz = idx / (x * y);

    //# get centroid label and save it to assignments
    int i, j, k;
    i = px * cx / x;
    j = py * cy / y;
    k = pz * cz / z;
    assignments[idx] = i + (j * cx) + (k * cx * cy);
}
""").get_function("first_assignments")

recompute_centroids_func = SourceModule(
"""
//# This code should be run with one thread per pixel (max img size is 4096x4096)
//# Responsible to updating pixel to superpixel assignments based on new centroids
__global__ void recompute_centroids(float* img, float* img_dim, float* cents, float* cents_dim, float* assignment) {

}""").get_function("recompute_centroids")
# pycuda tutorial from: https://documen.tician.de/pycuda/tutorial.html

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy

a = numpy.random.randn(4,4)
b = numpy.random.randn(4, 4, 2)



a = a.astype(numpy.float32)
b = b.astype(numpy.float32)
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

mod = SourceModule("""
  __global__ void doublify(float *a, float ***b)
  {
    int idx = threadIdx.x + threadIdx.y * blockDim.x;
    a[idx] *= 2;

    b[threadIdx.x][threadIdx.y][0] = (float) (threadIdx.x + 100.0 * threadIdx.y);
  }
  """)

func = mod.get_function("doublify")
func(a_gpu, b_gpu, block=(4,4,1))

a_doubled = numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)

print "initial a:"
print a
print "\na doubled:"
print a_doubled

b_doubled = numpy.empty_like(b)
cuda.memcpy_dtoh(b_doubled, b_gpu)

print "initial b:"
print b
print "\nb doubled:"
print b_doubled

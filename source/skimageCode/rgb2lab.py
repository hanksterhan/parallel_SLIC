import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from skimage.util import img_as_float

def doRGB2LABConversion():

    # read in the image into 3D array: height x width x 3 (rgb)
    image = img_as_float(io.imread("~/input/tiny.jpg"))

    height = image.shape[0]
    width = image.shape[1]

    # the lab vector is an empty vector of the same size as image
    # will store lab values instead of rgb values
    # TODO: how to pass in multiple values to pycuda function
    lab_vector = [[[0, 0, 0] for w in width] for h in height]


    # Allocate memory on GPU
    image_gpu = cuda.mem_alloc(image.nbytes)
    lab_gpu = cuda.mem_alloc(image.nbytes)

    # Copy data structure onto GPU
    cuda.memcpy_htod(image_gpu, image)
    cuda.memcpy_htod(lab_gpu, lab_vector)


    #TODO: check the inputs to doRGB2LABConv
    # PyCuda Code
    mod = SourceModule("""
        __global__
        void RGB2XYZ(
        	const int&		sR,
        	const int&		sG,
        	const int&		sB,
        	double&			X,
        	double&			Y,
        	double&			Z)
        {
        	double R = sR/255.0;
        	double G = sG/255.0;
        	double B = sB/255.0;

        	double r, g, b;

        	if(R <= 0.04045)	r = R/12.92;
        	else				r = pow((R+0.055)/1.055,2.4);
        	if(G <= 0.04045)	g = G/12.92;
        	else				g = pow((G+0.055)/1.055,2.4);
        	if(B <= 0.04045)	b = B/12.92;
        	else				b = pow((B+0.055)/1.055,2.4);

        	X = r*0.4124564 + g*0.3575761 + b*0.1804375;
        	Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
        	Z = r*0.0193339 + g*0.1191920 + b*0.9503041;
        }

        __global__
        void RGB2LAB(const int& sR, const int& sG, const int& sB, double& lval, double& aval, double& bval)
        {
        	//------------------------
        	// sRGB to XYZ conversion
        	//------------------------
        	double X, Y, Z;
        	RGB2XYZ(sR, sG, sB, X, Y, Z);

        	//------------------------
        	// XYZ to LAB conversion
        	//------------------------
        	double epsilon = 0.008856;	//actual CIE standard
        	double kappa   = 903.3;		//actual CIE standard

        	double Xr = 0.950456;	//reference white
        	double Yr = 1.0;		//reference white
        	double Zr = 1.088754;	//reference white

        	double xr = X/Xr;
        	double yr = Y/Yr;
        	double zr = Z/Zr;

        	double fx, fy, fz;
        	if(xr > epsilon)	fx = pow(xr, 1.0/3.0);
        	else				fx = (kappa*xr + 16.0)/116.0;
        	if(yr > epsilon)	fy = pow(yr, 1.0/3.0);
        	else				fy = (kappa*yr + 16.0)/116.0;
        	if(zr > epsilon)	fz = pow(zr, 1.0/3.0);
        	else				fz = (kappa*zr + 16.0)/116.0;

        	lval = 116.0*fy-16.0;
        	aval = 500.0*(fx-fy);
        	bval = 200.0*(fy-fz);
        }

        __global__
        void DoRGBtoLABConversion(
        	float*                      img,
        	gloat*                      lab)
        {
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

            int r = image[3 * idx + 0];
            int g = image[3 * idx + 1];
            int b = image[3 * idx + 2];

            RGB2LAB(r, g, b, lab[3 * idx + 0], lab[3 * idx + 1], lab[3 * idx + 2])

        }

    """)

    func = mod.get_function("DoRGBtoLABConversion")
    func(image_gpu, lab_gpu, block=(image.shape[1],image.shape[0],1)) # TODO: tweak the block sizes based on the width and height of the image

    image_superpixels = numpy.empty_like(image)
    cuda.memcpy_dtoh(image_superpixels, image_gpu)

    print(image_superpixels)
    print(image)

doRGB2LABConversion()

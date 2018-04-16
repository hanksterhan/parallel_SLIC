import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

"""
//==============================================================================
///	RGB2XYZ
///
/// sRGB (D65 illuninant assumption) to XYZ conversion
//==============================================================================
void SLIC::RGB2XYZ(
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
"""
def rgb2xyz(sR, sG, sB):
    R = sR / 255.0
    G = sG / 255.0
    B = sB / 255.0

    if R <= 0.04045:
        r = R / 12.92
    else:
        r = ((R + 0.055)/1.055) ** 2.4

    if G <= 0.04045:
        g = G / 12.92
    else:
        g = ((G + 0.055)/1.055) ** 2.4

    if B <= 0.04045:
        b = B / 12.92
    else:
        b = ((B + 0.055)/1.055) ** 2.4

	X = r*0.4124564 + g*0.3575761 + b*0.1804375
	Y = r*0.2126729 + g*0.7151522 + b*0.0721750
	Z = r*0.0193339 + g*0.1191920 + b*0.9503041

    return (X,Y,Z)
"""
//===========================================================================
///	RGB2LAB
//===========================================================================
void SLIC::RGB2LAB(const int& sR, const int& sG, const int& sB, double& lval, double& aval, double& bval)
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
"""
def rgb2lab(sR, sG, sB):
    # sRGB to XYZ conversion
    X = 0
    Y = 0
    Z = 0
    X, Y, Z = rgb2xyz(sR, sG, sB, X, Y, Z)

    # XYZ to LAB conversion
    epsilon = 0.008856 # actual CIE standard
    kappa   = 903.3 # actual CIE standard

    Xr = 0.950456 # reference white
    Yr = 1.0 # reference white
    Zr = 1.088754 # reference white

    xr = X / Xr
    yr = Y / Yr
    zr = Z / Zr

    if xr > epsilon:
        fx = xr ** (1.0 / 3.0)
    else:
        fx = (kappa * xr + 16.0) / 116.0

    if yr > epsilon:
        fy = yr ** (1.0 / 3.0)
    else:
        fy = (kappa * yr + 16.0) / 116.0

    if zr > epsilon:
        fz = zr ** (1.0 / 3.0)
    else:
        fz = (kappa * zr + 16.0) / 116.0

    lval = 116.0 * fy - 16.0
	aval = 500.0 * (fx - fy)
	bval = 200.0 * (fy - fz)

    return (lval, aval, bval)

"""
//===========================================================================
///	DoRGBtoLABConversion
///
///	For whole image: overlaoded floating point version
//===========================================================================
void SLIC::DoRGBtoLABConversion(
	const unsigned int*&		ubuff,
	double*&					lvec,
	double*&					avec,
	double*&					bvec)
{
	int sz = m_width*m_height;
	lvec = new double[sz];
	avec = new double[sz];
	bvec = new double[sz];

	for( int j = 0; j < sz; j++ ) // TODO: parallelize this part
	{
		int r = (ubuff[j] >> 16) & 0xFF;
		int g = (ubuff[j] >>  8) & 0xFF;
		int b = (ubuff[j]      ) & 0xFF;

		RGB2LAB( r, g, b, lvec[j], avec[j], bvec[j] );
	}
}
"""

# Need to double check how sci kit image reads in images
def doRGB2LABConversion(image):
    # is image [z,x,y,rgb] ?
    width = image[1]
    height = image[2]

    blockx = 4 # TODO: How many threads in the x direction?
    blocky = 4 # TODO: How many threads in the y direction?

    blockwidth = width / blockx
    blockheight = height /blocky

    """lvec = []
    avec = []
    bvec = []

    for j in range(size): # TODO: use pycuda for this
        r = image[j][0]
        g = image[j][1]
        b = image[j][2]

        l,a,b = rgb2lab(r, g, b)

        lvec.append(l)
        avec.append(a)
        bvec.append(b)"""

    # Allocate memory on GPU
    a_gpu = cuda.mem_alloc(image.nbytes)

    # Copy data structure onto GPU
    cuda.memcpy_htod(a_gpu, image)

    #TODO: check the inputs to doRGB2LABConv
    # PyCuda Code
    mod = SourceModule("""
        __global__
        void doRGB2LABConv(int** image, int width, int height, int blockwidth, int blockheight){
            //int idx = threadIdx.x + threadIdx.y * width // need to check how to assign a bucket(s) to each thread

            idx = blockIdx.x * blockDim.x + threadIdx.x // Actual index of the pixel to be processed

            start_x = blockwidth * threadIdx.x
            start_y = blockheight * threadIdx.y

            if(blockIdx.x == threadIdx.x){
                //TODO: check how to get the width of the block
                end_x = width
            } else{
                end_x = blockwidth * (threadIdx.x + 1)
            }

            if(blockIdx.y == threadIdx.y){
                //TODO: check how to get the height of the block
                end_y = height
            } else{
                end_y = blockheight * (threadIdx.y + 1)
            }

            for(start_y; start_y<=end_y; start_y++){
                for(start_x; start_x<=end_x; start_x++){
                    // Need to debug the next line to see if it actually produces the right numbers
                    idx = threadIdx.x * width + threadIdx.y * blockheight + start_y * width + start_x

                    r = image[idx][0]
                    g = image[idx][1]
                    b = image[idx][2]

                    l, a, b = rgv2lab(r, g, b)

                    lvec.append(l)
                    avec.append(a)
                    bvec.append(b)
                }
            }
        }

        __global__
        void rgb2xyz(
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
        void rgb2lab(const int& sR, const int& sG, const int& sB, double& lval, double& aval, double& bval)
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
    """)

    func = mod.get_function("doRGB2LABConv")
    # TODO: tweak the block sizes based on the width and height of the image
    func(a_gpu, block=(4,4,1))

    image_superpixels = numpy.empty_like(image)
    cuda.memcpy_dtoh(image_superpixels, a_gpu)

    print(image_superpixels)
    print(image)

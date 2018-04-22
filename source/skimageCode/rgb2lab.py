import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

def doRGB2LABConversion(image): # what is the format of image?
    # is image [z, x, y. rgb] ?
    width = image[1]
    width = image[2]

    blockx = 4 # TODO: tweak this for the number of blocks in the x direction
    blocky = 4 # TODO: tweak this for the number of blocks in the y direction

    # Are we having a 2D block grid each with a 2D thread grid??

    # Allocate memory on GPU
    a_gpu = cuda.mem_alloc(image.nbytes)

    # Copy data structure onto GPU
    cuda.memcpy_htod(a_gpu, image)

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

    """)

    func = mod.get_function("DoRGBtoLABConversion")
    func(a_gpu, block=(4,4,1)) # TODO: tweak the block sizes based on the width and height of the image

    image_superpixels = numpy.empty_like(image)
    cuda.memcpy_dtoh(image_superpixels, a_gpu)

    print(image_superpixels)
    print(image)

doRGB2LABConversion(image)

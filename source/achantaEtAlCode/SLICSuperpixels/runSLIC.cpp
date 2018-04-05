#include <string>
#include <iostream>
#include <fstream>
#include <sys/types.h>
#include "SLIC.h"

int readImage(int pixels[], int numColumns, int numRows);

int MAXIMGSIZE = 2000000;

int main()
{

	int width(0), height(0);
	// unsigned int (32 bits) to hold a pixel in ARGB format as follows:
	// from left to right,
	// the first 8 bits are for the alpha channel (and are ignored)
	// the next 8 bits are for the red channel
	// the next 8 bits are for the green channel
	// the last 8 bits are for the blue channel
	unsigned int* pbuff = new unsigned int[MAXIMGSIZE];
	//ReadImage(pbuff, width, height);//YOUR own function to read an image into the ARGB format

  int pixels[MAXIMGSIZE];
  readImage(pixels, width, height);

	//----------------------------------
	// Initialize parameters
	//----------------------------------
	int k = 200;//Desired number of superpixels.
	double m = 20;//Compactness factor. use a value ranging from 10 to 40 depending on your needs. Default is 10
	int* klabels = NULL;
	int numlabels(0);
	string filename = "yourfilename.jpg";
	string savepath = "yourpathname";

  //----------------------------------
	// Perform SLIC on the image buffer
	//----------------------------------
	SLIC segment;
	segment.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(pbuff, width, height, klabels, numlabels, k, m);
	// Alternately one can also use the function DoSuperpixelSegmentation_ForGivenStepSize() for a desired superpixel size

  //----------------------------------
	// Save the labels to a text file
	//----------------------------------
	segment.SaveSuperpixelLabels(klabels, width, height, filename, savepath);

  //----------------------------------
	// Draw boundaries around segments
	//----------------------------------
	segment.DrawContoursAroundSegments(pbuff, klabels, width, height, 0xff0000);

  //----------------------------------
	// Save the image with segment boundaries.
	//----------------------------------
  //TODO: code to save segmented img
	//SaveSegmentedImageFile(pbuff, width, height);//YOUR own function to save an ARGB buffer as an image

  //----------------------------------
	// Clean up
	//----------------------------------
	if(pbuff) delete [] pbuff;
	if(klabels) delete [] klabels;

	return 0;
}

int readImage(int pixels[], int numColumns, int numRows){
  string inputFileName, outputFileName; // holds name of input and output
                                        // image files
  // file header info
  string imgFormat;
  int maxColorValue;

  // Greet user and ask for input and output image file names, storing
  // as the variable inputFileName and outputFileName respectively
  cout << endl << "Welcome to SLIC" << endl;
  cout << endl << "Enter name of image file: ";
  cin >> inputFileName;
  cout << "Enter name of output file: ";
  cin >> outputFileName;

  // Open and instream and an outstream
  ifstream inFile;                          // stream for reading from file
  // ofstream outFile(outputFileName.c_str()); // stream for writing to file

  // Open the image speficied by the user, and check that it worked
  inFile.open(inputFileName.c_str());
  if(!inFile.is_open()){
    cout << "Error: Invalid file name." << endl;
    return 1;
  }

  // Read off the first four pieces of data (the image header)
  inFile >> imgFormat;
  inFile >> numColumns;
  inFile >> numRows;
  inFile >> maxColorValue;

  // Make sure the image is not too large for the buffer to handle a whole line
  if(numRows * numColumns * 3 > MAXIMGSIZE){
    cout << "Error: Image is too large." << endl;
    return 1;
  }

  // Read in file
  // while(!inFile.eof()){
  for(int i = 0; i < numRows * numColumns * 3; i++){
    inFile >> pixels[i];
  }
  inFile.close();
  return 0;
}

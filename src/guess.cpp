#include "GuessCaptcha.h"
#include <string>
#include <stdio.h>
#include "tclap/CmdLine.h"

using namespace std;
using namespace TCLAP;

int main(int argc, char** argv) {
	GuessCaptcha gc;
	
	try {
		// Define and parse the command line arguments.
		CmdLine cmd("Captcha reading utility.", ' ', "0.1");
		ValueArg<string> inputFileArg("f", "file", "The file containing the captcha image the script will attempt to read and guess what it says.", true, "", "string path/filename", cmd);
		ValueArg<string> trainingSource("t", "trainingSource", "The directory containing all the images to train the NN against. Should contain a directory of images whose name matches the character it depicts, ie 'F.gif'.", true, "", "string path", cmd);
		ValueArg<int> segmentValueArg("s", "segmentValue", "Value used to segment the image.", false, 1, "int", cmd);
		ValueArg<int> whiteThresholdArg("w", "whiteThreshold", "Value used to white threshold the image.", false, 20 ,"int", cmd);
		ValueArg<int> slicePixelArg("p", "slice", "The pixel on which this image will be sliced.", false, 35, "int", cmd);
		cmd.parse(argc, argv);
		
		// Set the instance variables to the command line arguments.
		gc.inputFile = inputFileArg.getValue();
		gc.trainingSource = trainingSource.getValue();
		gc.segmentValue = segmentValueArg.getValue();
		gc.whiteThreshold = whiteThresholdArg.getValue();
		gc.slicePixel = slicePixelArg.getValue();
		
	} catch (ArgException &e) {
		cerr << "Error: " << e.error() << " for arg " << e.argId() << endl;
	}
	
	// Start once all the instance variables have been assigned.
	gc.start();
	printf("I think the file you gave me says %s.\n", gc.getGuess().c_str());
	
	// testing
	//printf("inputFile = %s\nsegmentValue = %d\nwhiteThreshold = %d\nslicePixel = %d\n", gc->inputFile.c_str(), gc->segmentValue, gc->whiteThreshold, gc->slicePixel);
}

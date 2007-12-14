#ifndef GUESSCAPTCHA_H_
#define GUESSCAPTCHA_H_

#include <string>
#include <vector>

class GuessCaptcha {
public:
	GuessCaptcha();
	
	/**
	 * Segments the image with imagemagick by using the command line and
	 * creating a new image. Segmentation values can be adjusted using
	 * the related instance variables.
	 */
	void segmentImage();
	/**
	 * Slices the image based, to begin it will comply cut on pixel values.
	 */
	void sliceImage();
	void resizeSlices();
	void readPixels();
	void builtNN();
	void computeData();
	void readOutputs();
	std::string getGuess();
	
	int segmentValue; /**< The value passed to convert specified by -convert segmentValue. */
	int whiteThreshold; /**< The value passed to convert specified by -white-threshold whiteThreshold. */
	int numSlices; /**< The number of slices created. */
	int slicePixel; /**< The pixel interval on which the image will be sliced. */
	
	std::string inputFile; /**< The original file location passed in from the command line. */
	std::string segmentedImageLocation; /**< The file location of the newly segmented image. */
	
	std::vector<std::string> slicedImagesLocations; /**< The locations of the sliced images. */
	std::vector<std::vector<double> > pixelValues; /**< Contains all the pixel values of each of the sliced images. */
};

#endif /*GUESSCAPTCHA_H_*/

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
	 * Slices the image; to begin it will simply cut on pixel values.
	 */
	void sliceImage();
	/**
	 * Resize each of the slices to the desired size of the neural network's input.
	 * This should be fixed, as training data must also conform to this dimension.
	 */
	void resizeSlices();
	/**
	 * Read all the pixel values into a data structure for input into the neural
	 * network. Keeping each picture distinct in its own data structure is necessary.
	 */
	void readPixels();
	/**
	 * Build and initialize the neural network, including training using the
	 * desired training set.
	 */
	void builtNN();
	/**
	 * Compute the data from the pixel values associated with an image. This
	 * must be done n times with n slices.
	 * @param data The pixel values for one image slice.
	 */
	void computeData(std::vector<double> data);
	/**
	 * Read the outputs following a computation of the data. This will interpret
	 * the values of the output layer into an alphanumeric character and store
	 * it in \doxygen{guess}.
	 */
	void readOutputs();
	void start();
	std::string getGuess();
	
	int segmentValue; /**< The value passed to convert specified by -convert segmentValue. */
	int whiteThreshold; /**< The value passed to convert specified by -white-threshold whiteThreshold. */
	int numSlices; /**< The number of slices created. */
	int slicePixel; /**< The pixel interval on which the image will be sliced. */
	
	std::string guess; /**< Stores the accumulated guesses for each of the data sets that have been passed to \doxygen{computeData}. */
	std::string inputFile; /**< The original file location passed in from the command line. */
	std::string segmentedImageLocation; /**< The file location of the newly segmented image. */
	
	std::vector<std::string> slicedImagesLocations; /**< The locations of the sliced images. */
	std::vector<std::vector<double> > pixelValues; /**< Contains all the pixel values of each of the sliced images. */
};

#endif /*GUESSCAPTCHA_H_*/

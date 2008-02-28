#ifndef GUESSCAPTCHA_H_
#define GUESSCAPTCHA_H_

#include <string>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <omp.h>
#include <sstream>
#include "ImageMagick/Magick++.h"
#include "NeuralNet.h"

class GuessCaptcha {
public:
	GuessCaptcha();
	
	/**
	 * Segments the image with imagemagick by using the command line and
	 * creating a new image. Segmentation values can be adjusted using
	 * the related instance variables. In the same step the image is
	 * also sliced on the pixels interval specified by slicePixel.
	 */
	void segmentImage() throw(char*);
	/**
	 * Resize each of the slices to the desired size of the neural network's input.
	 * This should be fixed, as training data must also conform to this dimension.
	 */
	void resizeSlices() throw(char*);
	/**
	 * Read all the pixel values into a data structure for input into the neural
	 * network. Keeping each picture distinct in its own data structure is necessary.
	 */
	void readPixels() throw(char*);
	/**
	 * Build and initialize the neural network, including training using the
	 * desired training set.
	 */
	void buildNN();
	/** 
	 * Compute the data from the pixel values associated with an image.
	 */
	void computeData();
	/**
	 * Read the outputs following a computation of the data. This will interpret
	 * the values of the output layer into an alphanumeric character and store
	 * it in guess.
	 */
	void readOutputs();
	/**
	 * This will start all the of the processing that needs to be done before
	 * the getGuess() method can be called.
	 * @pre The required instance variables have all been assigned.
	 * @post The instance variable guess now contains the system's best guess of the characters in the original captcha image.
	 */
	void start();
	/**
	 * Returns the string that the system determined that was in the original captcha.
	 * @return The accumulated guesses for each of the data sets that have been passed to computeData.
	 */
	std::string getGuess();
	
	// collection of instance variables specified by the user
	int segmentValue; /**< The value passed to convert specified by -convert segmentValue. */
	int whiteThreshold; /**< The value passed to convert specified by -white-threshold whiteThreshold. */
	int numSlices; /**< The number of slices created. */
	int slicePixel; /**< The pixel interval on which the image will be sliced. */
	std::string inputFile; /**< The original file location passed in from the command line. The only required parameter. */

private:
	std::string segmentedImageLocation; /**< The file location of the newly segmented image. */
	std::string guess; /**< Stores the accumulated guesses for each of the data sets that have been passed to computeData. */
	std::vector<std::string> slicedImagesLocations; /**< The locations of the sliced images. */
	std::vector<std::vector<double> > pixelValues; /**< Contains all the pixel values of each of the sliced images. */
	std::string CHARACTER_MAP; /** Contains the mapping used to map the neurons to actual characters. */
	
	NeuralNet NN; /**< The neural network that will be used to guess each letter in the sliced images. */
	
	static const int NUM_INPUT_NEURONS = 1600; /**< Going to be using 40x40 images, so that means lots of input neurons. */
	static const int NUM_HIDDEN_NEURONS = 500; /**< Set a number of hidden neurons, this must be tweaked. */
	static const int NUM_OUTPUT_NEURONS = 62; /**< The number of output neurons is determined by 26(lower case)+26(upper case)+10(numbers) = 62. */
};

#endif /*GUESSCAPTCHA_H_*/

#include "GuessCaptcha.h"
#include "Random.h"

GuessCaptcha::GuessCaptcha() {
	this->guess.clear();
	this->CHARACTER_MAP = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
}

void GuessCaptcha::buildNN() {
	NeuralNet nn(this->NUM_INPUT_NEURONS, this->NUM_HIDDEN_NEURONS, this->NUM_OUTPUT_NEURONS);
	this->NN = nn;
}

std::string GuessCaptcha::getGuess() {
	return this->guess;
}

void GuessCaptcha::start() {
	// lots-o-functions
	try {
		this->buildNN();
		this->train();
		this->segmentImage();
		this->resizeSlices();
		this->readPixels();
		this->computeData();
		this->readOutputs();
	} catch (char* e) {
		std::printf("Exception raised: %s\n", e);
	} catch (std::exception e) {
		std::printf("Exception raised: %s\n", e.what());
	}
}

std::string generateRandomString(int len) {
	std::string chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_";
	int max = chars.size();
	Random r;
	std::string res("", len);
	
	for (int i=0; i<len; ++i ) {
		res.at(i) = chars.at(r.strong_range(max));
	}
	
	return res;
}

void GuessCaptcha::segmentImage() throw(char*) {
	int slices, cropXStart, cropXEnd;
	std::string randomName;
	Magick::Geometry sliceGeo(40, 40);
	Magick::Image image;
	try {
		image.read(this->inputFile);
	}
	catch (...) {
		std::cerr << "Couldnt read " << this->inputFile << "." << std::endl;
		return;
	}
	int width = (int) image.columns();
	
	// segment and threshold out the noise
//	image.segment(this->segmentValue);
//	image.threshold(this->whiteThreshold);
	// set to gray after we segment and treshold the image
	// otherwise bad things happen, like big grey blobs,
	// and noone likes a big grey blob, unless you are a
	// big grey blob, but even then you might be
	// a little self conscious
	image.type(Magick::BilevelType);
	image.modifyImage();
	
	// now to crop into slices
	// Get the number of slices to make which is the ceiling
	// of width/width_of_slices.
//	slices = (int) ceil((double)width/this->slicePixel);
//	for (int i=0; i<slices; ++i) {
		Magick::Image imgCopy(image);
//		// pictorially this looks like:
//		// [----]------------------------
//		//  ----[----]--------------------
//		//  ---------[----]---------------
//		//  ...
//		//  -------------------------[---]
//		// if the slicePixel were 4.
//		cropXStart = this->slicePixel * i;
//		cropXEnd = this->slicePixel + cropXStart;
//		if (cropXEnd > width) cropXEnd = width-1;
//		std::cout << "cropping from " << cropXStart << " to " << cropXEnd << endl;
//		imgCopy.modifyImage();
//		imgCopy.chop(Magick::Geometry(cropXStart, 0));
//		imgCopy.crop(Magick::Geometry(imgCopy.rows(), cropXEnd));
//		
//		// might as well resize it too, and deprecate resizeSlices()
		imgCopy.scale(sliceGeo);
//		
//		// write it out to a randomly generated name
//		// and keep track of it in our member variable
//		// saving it as a pbm, an X representation of black
//		// and white images
		randomName = "/tmp/" + generateRandomString(10) + ".pbm";
		this->slicedImagesLocations.push_back(randomName);
		imgCopy.write(randomName);
//		imgCopy.display();
//	}
}

void GuessCaptcha::resizeSlices() throw(char*) {
	// deprecated, boo go away
	return;
}

void GuessCaptcha::readPixels() throw(char*) {
	int width, height;
	int num_slices = this->slicedImagesLocations.size();
	std::vector<double> tmpImage;
	
	// dont want any mishaps
	this->pixelValues.empty();
	
//	#pragma omp parallel for
	for (int i=0; i<num_slices; ++i) {
		Magick::Image image(this->slicedImagesLocations.at(i));
		image.type(Magick::BilevelType);
		width = (int) image.columns();
		height = (int) image.rows();
		tmpImage.resize(width*height, 0);
			
		// move the image data into the pixel values structure
		for (int h=0; h<height; ++h) {
			for (int w=0; w<width; ++w) {
				tmpImage.push_back((double)(((Magick::ColorMono)image.pixelColor(w, h)).mono()));
			}
		}
		this->pixelValues.push_back(tmpImage);
	}
}

void GuessCaptcha::computeData() {
	int numArrays = this->pixelValues.size();
	for (int i=0; i<numArrays; ++i) {
		this->NN.inputData = this->pixelValues.at(i);
		this->NN.compute();
	}
}

void GuessCaptcha::readOutputs() {
	size_t largestIndex = 0;
	double largestValue = 0;
	std::string tmpGuess;
	this->guess.clear();
	
	// get the largest index which we will use to map to a character.
//	#pragma omp parallel for
	for (int i=0; i<this->NN.numOutput; ++i) {
		if (this->NN.output.neurons.at(i).value > largestValue) {
			largestIndex = i;
			largestValue = this->NN.output.neurons.at(i).value;
		}
	}
	
	// map the index with the largest value to a character.
	if (largestIndex >= this->CHARACTER_MAP.size() ) throw("Got an invalid index from readOutputs().");
	tmpGuess = this->CHARACTER_MAP.at(largestIndex);
	this->guess.append(tmpGuess);
}

void GuessCaptcha::train() {
	NNTrainer trainer(this->NN, this->trainingSource);
	trainer.train();
}

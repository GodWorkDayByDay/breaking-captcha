#include "GuessCaptcha.h"
#include <png++/png.hpp>

GuessCaptcha::GuessCaptcha() {
	this->guess = "something";
	this->CHARACTER_MAP = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
}

void GuessCaptcha::buildNN() {
	NeualNet nn(this->NUM_INPUT_NEURONS, this->NUM_HIDDEN_NEURONS, this->NUM_OUTPUT_NEURONS));
	this->NN = nn;
	//TODO: training data, no idea
}

std::string GuessCaptcha::getGuess() {
	return this->guess;
}

void GuessCaptcha::start() {
	// lots-o-functions
	try {
		this->segmentImage();
		this->resizeSlices();
		this->readPixels();
		this->buildNN();
		this->computeData();
	} catch (char* e) {
		std::printf("Exception raised: %s\n", e);
	}
}

std::string itos(int i) {
	std::stringstream s;
	s << i;
	return s.str();
}

void GuessCaptcha::segmentImage() throw(char*) {
	// magic command to segment, slice and resize
	// convert -segment 20 -white-threshold 20 -crop 29 -depth 1 c1.gif c1.png
	// mogrify -resize 50x50 c1-*.png
	
	int slices;
	std::string cmd;
	png::image<png::gray_pixel_1> image(this->inputFile);
	int width = (int) image.get_width();
	int height = (int) image.get_height();
	
	cmd = cmd + "convert -segment " + itos(this->segmentValue);
	cmd = cmd + " -white-threshold " + itos(this->whiteThreshold);
	cmd = cmd + " -crop " + itos(this->slicePixel) + " -depth 1 " + this->inputFile + " /tmp/c1.png";
	
	if (!std::system("rm -fR /tmp/c1-*.png") || !std::system(cmd.c_str())) {
		throw "Invalid command.";
	}
	
	// Get the number of slices we made which is the ceiling
	// of width/width_of_slices.
	slices = (int) ceil((float)width/this->slicePixel);
	for (int i=0; i<slices; ++i) {
		this->slicedImagesLocations.push_back("/tmp/c1-"+itos(i)+".png");
	}
}

void GuessCaptcha::resizeSlices() throw(char*) {
	std::string cmd;
	int size = this->slicedImagesLocations.size();
	
	for (int i=0; i<size; ++i) {
		cmd = "mogrify -resize 40x40 " + this->slicedImagesLocations.at(i);
		
		if (!std::system(cmd.c_str())) {
			throw "Invalid command.";
		}
	}
}

void GuessCaptcha::readPixels() throw(char*) {
	unsigned long rowBytes, width, height;
	int num_slices = this->slicedImagesLocations.size();
	std::vector<double> tmpImage;
	
	// dont want any mishaps
	this->pixelValues.empty();
	
	for (int i=0; i<num_slices; ++i) {
		// open a file pointer for readpng to read
		png::image<png::gray_pixel_1> image(this->slicedImagesLocations.at(i));
		width = image.get_width();
		height = image.get_height();
			
		// move the image data into the pixel values structure
		for (int h=0; h<height; ++h) {
			for (int w=0; w<width; ++w) {
				tmpImage.push_back(image.get_pixel(w, h));
			}
		}
		this->pixelValues.push_back(tmpImage);
		
		// cleanup the mess
		readpng_cleanup(1);
		fclose(sliceFile);
	}
}

void GuessCaptcha::computeData() {
	int numArrays = this->pixelValues.size();
	for (int i=0; i<numArrays; ++i) {
		this->NN.inputData = this->pixelValues.at(i);
		this->NN.compute();
		this->readOutputs();
	}
}

void GuessCaptcha::readOutputs() {
	int largestIndex = 0;
	double largestValue = 0;
	
	// get the largest index which we will use to map to a character.
	#pragma omp parallel
	for (int i=0; i<this->NN.numOutput; ++i) {
		if (this->NN.output.neurons.at(i).value > largestValue) {
			largestIndex = i;
			largestValue = this->NN.output.neurons.at(i).value;
			break;
		}
	}
	
	// map the index with the largest value to a character.
	this->guess.append((const char*) this->CHARACTER_MAP[largestIndex]);
}

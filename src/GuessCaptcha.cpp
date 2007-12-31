#include "GuessCaptcha.h"
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <math.h>
#include <omp.h>
#include <sstream>
#include "NeuralNet.h"
#include "readpng.h"

GuessCaptcha::GuessCaptcha() {
	this->guess = "something";
	this->CHARACTER_MAP = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
}

void GuessCaptcha::buildNN() {
	this->NN = new NeuralNet(this->NUM_INPUT_NEURONS, this->NUM_HIDDEN_NEURONS, this->NUM_OUTPUT_NEURONS);
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
	
	unsigned long width, height;
	int slices, res;
	std::string cmd;
	FILE* wholeFile = fopen(this->inputFile.c_str(), "rb");
	
	cmd = cmd + "convert -segment " + itos(this->segmentValue);
	cmd = cmd + " -white-threshold " + itos(this->whiteThreshold);
	cmd = cmd + " -crop " + itos(this->slicePixel) + " -depth 1 " + this->inputFile + " /tmp/c1.png";
	
	if (!std::system("rm -fR /tmp/c1-*.png") || !std::system(cmd.c_str())) {
		throw "Invalid command.";
	}
	
	res = readpng_init(wholeFile, &width, &height);
	if (res == 0) {
		// Get the number of slices we made which is the ceiling
		// of width/width_of_slices.
		slices = (int) ceil((float)width/(float)this->slicePixel);
		for (int i=0; i<slices; ++i) {
			this->slicedImagesLocations.push_back("/tmp/c1-"+itos(i)+".png");
		}
		
		//cleanup our mess
		fclose(wholeFile);
		readpng_cleanup(0);
	}
	else {
		fclose(wholeFile);
		readpng_cleanup(0);
		throw("Error reading file.");
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
	int channels;
	int initRes, num_slices = this->slicedImagesLocations.size();
	unsigned char* image_data;
	double* image_info;
	FILE* sliceFile;
	
	// Since I don't feel like rewriting the code from the png book,
	// I'm just going to typecast the chars to doubles.
	
	for (int i=0; i<num_slices; ++i) {
		// open a file pointer for readpng to read
		sliceFile = fopen(this->slicedImagesLocations.at(i).c_str(), "r");
		
		// init and read the png, otherwise throw an error
		initRes = readpng_init(sliceFile, &width, &height);
		if ( initRes==0 ) {
			image_data = readpng_get_image(0, &channels, &rowBytes);
			
			// move the char* data into the double*
			for (int j=0; j<image_data; ++j) {
				image_info[j] = (double) image_data[j];
			}
			this->pixelValues.push_back(image_info);
			
			// cleanup the mess
			readpng_cleanup(1);
			fclose(sliceFile);
		}
		else {
			readpng_cleanup(1);
			fclose(sliceFile);
			throw("Error getting image pixels.");
		}
	}
}

void GuessCaptcha::computeData() {
	int numArrays = this->pixelValues.size();
	for (int i=0; i<numArrays; ++i) {
		this->NN->inputData = this->pixelValues[i];
		this->NN->compute();
		this->readOutputs();
	}
}

void GuessCaptcha::readOutputs() {
	int largestIndex = 0;
	double largestValue = 0;
	
	// get the largest index which we will use to map to a character.
	#pragma omp parallel
	for (int i=0; i<this->NN->numOutput; ++i) {
		if (this->NN->output->neurons[i].value > largestValue) {
			largestIndex = i;
			largestValue = this->NN->output->neurons[i].value;
			break;
		}
	}
	
	// map the index with the largest value to a character.
	this->guess.append((const char*) this->CHARACTER_MAP[largestIndex]);
}

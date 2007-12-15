#include "GuessCaptcha.h"
#include <string>
#include <vector>
#include "NeuralNet.h"

GuessCaptcha::GuessCaptcha() {
	this->guess = "something";
	this->CHARACTER_MAP = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
}

void GuessCaptcha::buildNN() {
	this->NN = new NeuralNet(this->NUM_INPUT_NEURONS, this->NUM_HIDDEN_NEURONS, this->NUM_OUTPUT_NEURONS);
	//TODO: training data
}

std::string GuessCaptcha::getGuess() {
	return this->guess;
}

void GuessCaptcha::start() {
	// lots-o-functions
	this->segmentImage();
	this->sliceImage();
	this->resizeSlices();
	this->readPixels();
	this->buildNN();
	this->computeData();
}

void GuessCaptcha::segmentImage() {
	//TODO: impelment segmentation
}

void GuessCaptcha::sliceImage() {
	//TODO: see if i can do this in one big step with imagemagick
}

void GuessCaptcha::resizeSlices() {
	//TODO: resize command lines
}

void GuessCaptcha::readPixels() {
	//TODO: not sure how to read in pixel values yet.
}

void GuessCaptcha::computeData() {
	for (int i=0; i<this->pixelValues.size(); ++i) {
		this->NN->inputData = this->pixelValues[i];
		this->NN->compute();
		this->readOutputs();
	}
}

void GuessCaptcha::readOutputs() {
	// get the largest index which we will use to map to a character.
	int largestIndex = 0;
	double largestValue = 0;
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

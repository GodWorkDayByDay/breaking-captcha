/**
 * @file NeuralNet.cpp
 * @author Ben Snider
 * @version 0.1
 * 
 * Implements the NeuralNet class.
**/

#include "NeuralNet.h"
#include <cstdlib>

/**
 * @param numInput The number of neurons to create in the input layer.
 * @param numHidden The number of neurons to create in the hidden layer.
 * @param numOutput The number of neurons to create in the output layer.
**/
NeuralNet::NeuralNet(int numInput, int numHidden, int numOutput) {
	// assign member variables
	this->numInput = numInput;
	this->numHidden = numHidden;
	this->numOutput = numOutput;
	
	// instanciate the layers
	this->input = new GenericLayer(this->numInput, NULL, this->hidden);
	this->hidden = new GenericLayer(this->numHidden, this->input, this->output);
	this->output = new GenericLayer(this->numOutput, this->hidden, NULL);
	
	// init the layers
	this->input->init();
	this->hidden->init();
	this->output->init();
}

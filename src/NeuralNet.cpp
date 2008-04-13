/**
 * @file NeuralNet.cpp
 * @author Ben Snider
 * @version 0.1
 * 
 * Implements the NeuralNet class.
**/

#include "NeuralNet.h"
#include <iostream>

NeuralNet::NeuralNet(int numInput, int numHidden, int numOutput) {
	this->learningRate = 0.25;
	this->maxTrainingIterations = 1000;
	this->percentChange = 0.1;
	
	// assign member variables
	this->numInput = numInput;
	this->numHidden = numHidden;
	this->numOutput = numOutput;
	
	// instanciate the layers
	this->input.setNumNeurons(this->numInput);
	this->input.setChild(this->hidden);
	this->hidden = GenericLayer(this->numHidden, this->input, this->output);
	this->output.setNumNeurons(this->numOutput);
	this->output.setParent(this->hidden);
	
	// init the layers
	this->input.init();
	this->hidden.init();
	this->output.init();
}

NeuralNet::NeuralNet() {
	this->learningRate = 0.25;
	this->maxTrainingIterations = 1000;
	this->percentChange = 0.01;
	
	this->numInput = 0;
	this->numHidden = 0;
	this->numOutput = 0;
}

void NeuralNet::train() {
	int iteration=0;
	double currentMSE, lastMSE;
	currentMSE = 10;
	lastMSE = this->percentChange;
	
	// either stop training when we have reached a maximum acceptable number of iterations
	// or when the actual and desired responses are significantly close based on the rate of change
	while ( ((fabs((currentMSE-lastMSE)/currentMSE)) > this->percentChange) and (iteration < this->maxTrainingIterations) ) {
		this->compute();
		
		this->alterWeights(this->output);
		this->alterWeights(this->hidden);
		
		++iteration;
		lastMSE = currentMSE;
		currentMSE = this->calculateMSE();
	}
}

double NeuralNet::calculateMSE() {
	double sum=0;
	
//	#pragma omp parallel for reduction(+:sum)
	for (int i=0; i<this->output.numNeurons; ++i) {
		sum += pow(this->output.neurons.at(i).error, 2);
	}
	return sum/(double)this->output.numNeurons;
}

void NeuralNet::compute() {
	this->calculateNeuronValues(this->input);
	this->calculateNeuronValues(this->hidden);
	this->calculateNeuronValues(this->output);
}

std::vector<Neuron> NeuralNet::getOutput() {
	return this->output.neurons;
}

void NeuralNet::calculateNeuronValues(GenericLayer& layer) {
	double neuronValue(0);
	int j(0);
	
	// simply put the inputData into the input layer
	if ( !layer.hasParent ) {
//		#pragma omp parallel for if(layer.numNeurons > 500)
		for (int i=0; i<layer.numNeurons; ++i) {
			layer.neurons.at(i).value = this->inputData.at(i);
		}
		return;
	}
	
//	#pragma omp parallel for private(j) if(layer.numNeurons > 500)
	for (int i=0; i<layer.numNeurons; ++i) {
		neuronValue = 0;
		for (j=0; j<layer.parentLayer->numNeurons; ++j) {
			neuronValue += layer.parentLayer->neurons.at(j).value * layer.parentLayer->weights.at(j).at(i);
		}
		// addition of the bias term
		neuronValue += layer.neurons.at(i).bias * layer.neurons.at(i).biasWeight;
		
		// pass the value through the activation function
		neuronValue = this->logisticActivation(neuronValue);
		
		// the output layer's neurons are fired when the value is >0.5
//		if ( !layer.hasChild ) {
//			if ( neuronValue > 0.5 ) {
//				layer.neurons.at(i).value = 1;
//			}
//			else {
//				layer.neurons.at(i).value = 0;
//			}
//		}
	}
}

double NeuralNet::logisticActivation(double x) {
	return 1.0/(1.0 + exp(-x));
}

void NeuralNet::alterWeights(GenericLayer& layer) {
	int i;
	// do nothing if we for some reason try to alter the weights of the input layer
	if ( !layer.hasParent ) return;
	
	// the case for the output layer is simply the desired-actual values.
	else if ( !layer.hasChild ) {
//		#pragma omp parallel for if(layer.numNeurons > 500)
		for (i=0; i<layer.numNeurons; ++i) {
			// error = desired - actual
			layer.neurons.at(i).error = this->desiredOutput.at(i) - layer.neurons.at(i).value;
			layer.neurons.at(i).localGradient = layer.neurons.at(i).error * layer.neurons.at(i).value; 
		}
		// we uncouple this loop from the previous to access the weight array in row-major order
//		#pragma omp parallel for private(i) if(layer.parentLayer->numNeurons > 500)
		for (int j=0; j<layer.parentLayer->numNeurons; ++j) {
			for (i=0; i<layer.numNeurons; ++i) {
				layer.parentLayer->weights.at(j).at(i) += this->learningRate * layer.neurons.at(i).localGradient * layer.parentLayer->neurons.at(j).value;
			}
		}
	}
	
	// the case for the hidden layer is the back propogation algorithm
	else {
		double sumGradientWeights;
		
		int i(0), k(0);
		// loop to calculate all of the new weights from the parent layer to this layer (i to j)
//		#pragma omp parallel for private (k,i) if (layer.numNeurons > 100)
		for (int j=0; j<layer.numNeurons; j++) {
			sumGradientWeights = 0;
			// summing up the gradients of the child layer and the weights connecting them
			for (k=0; k<layer.childLayer->numNeurons; ++k) {
				sumGradientWeights += layer.childLayer->neurons.at(k).localGradient * layer.weights.at(j).at(k);
			}
			// now that we have the sum of those gradients we can calculate this neuron's gradient
			layer.neurons.at(j).localGradient = layer.neurons.at(j).value * (1-layer.neurons.at(j).value) * sumGradientWeights;
			
			// now that we have the local gradient we can change the weights between this neuron and those neurons connected to it in the parent layer
			for (i=0; i<layer.parentLayer->numNeurons; ++i) {
				layer.parentLayer->weights.at(i).at(j) += this->learningRate * layer.neurons.at(j).localGradient * layer.parentLayer->neurons.at(i).value;
			}
		}
	}
}



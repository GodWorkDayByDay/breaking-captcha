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

/**
 * @pre The trainingData and desiredValues arrays have been assigned and are of the appropriate dimensions for this particular NN.
 * @post All the weights and biases are set according to the trainingData and the NN is ready for real environment data. 
**/
void NeuralNet::train() {
	
}

/**
 * @pre The NN has been initialized and trained, and the inputData array has been assigned and is of the appropriate dimensions for this particular NN.
 * @post The output layer nodes now contain, based on the NN's weights and biases, the output values for the given data set.
**/
void NeuralNet::compute() {
	
}

/**
 * To calculate the value of each neuron we calculate the sum of the weights connected to each
 * neuron multiplied by the value of each corresponding neuronal value, finally adding the
 * adjusted bias and passing this value through an activation function.
 * 
 * The formula we use is given as follows, where \f$y_i\f$ is the i-th neuron on the parent
 * layer and \f$w_{ji}\f$ is the weight from the parent to the neuron whose value we are
 * calculating (note that in this notation, for simplicity, we denote the bias and its weight
 * as the first element): 
 * \f$\displaystyle v_j(n) = \sum_{i=0}^n y_i(n)w_{ji}(n)\f$
 * 
 * And similarly we note that the final value of this neuron is \f$y(n)_j = \phi_j(v_j(n))\f$
 * where \f$\phi_j()\f$ is the activation function on neuron j.
**/
void NeuralNet::calculateNeuronValues(GenericLayer* layer) {
	if ( layer->parentLayer == NULL ) return;
	double tmp=0;
	
	#pragma omp parallel
	for (int i=0; i<layer->numNeurons; ++i) {
		#pragma omp parallel
		for (int j=0; j<layer->parentLayer->numNeurons; ++j) {
			// sum(n(j)*w(j,i))
			tmp += layer->parentLayer->neurons[j].value * layer->parentLayer->weights[j][i];
		}
		// addition of the bias term, wich is essentially the j+1 weight on i
		tmp += layer->parentLayer->neurons[i].bias * layer->parentLayer->neurons[i].biasWeight
		
		// if this is the output layer then we just pass the value through a linear activation
		// function, ie y(i) = x.
		if ( layer->childLayer == NULL ) {
			layer->neurons[i] = tmp;
		}
		// otherwise we put it through the activation function of our choosing, in this case a
		// logistic one that is easily differentiated
		else {
			layer->neurons[i] = layer->logisticActivation(tmp);
		}
		tmp = 0;
	}
}

/**
 * Used for computing the value of a neuron after calculating the raw value.
 * We use an easily differentiable function so that it is easier later on to calculate the
 * change in weights. The particular function we use here is the sigmoidal activation
 * function, which constrains the output to between 0 and +1, a good fit for NNs.
 * 
 * The precise function we use is \f$\displaystyle\phi_j(x) = \frac{1}{1 + exp(-x)}\f$.
**/
double NeuralNet::logisticActivation(double x) {
	return 1.0/(1.0 + exp(-x));
}

/**
 * We calculate the errors of the each of the neurons in the NN here for use in adjusting
 * their weights and biases for training purposes. There are three main cases for a
 * three layer feed forward NN. 
 * 
 * The first is the input layer, and since the desired value is
 * always equal to the actual value, since it is given, the gradient is 0.
 * 
 * The second is the case of a hidden layer, where the errors are calculated from the layer nearest
 * to the output layer to the layer closest to the input layer. This is done using the
 * back propogation algorithm, which, given the gradients of adjacent layers, calculates
 * the gradients recursively working backwords and using the derivative of the activation
 * function used for calculating neuron values. This is done since the error
 * signals cannot be determined for hidden layers since there is no value to
 * compare their output to. It can be written as
 * \f$\displaystyle\gamma_j(n) = \phi_j^`(v_j(n))\sum_{k=0}^m \gamma_k(n)w_{kj}(n)\f$ where
 * neuron j is the gradient we are calculating, neuron k is in the child layer and m is
 * the number of neurons on that layer.
 * 
 * The third case is the output layer, where the error is trivially desired-actual. The
 * gradient is then defined much the same as case 2 where it is the error multiplied
 * by the derivative of the activation function applied to the value of the neuron. It
 * can be written as \f$\displaystyle\gamma_j(n) = e_j(n)\phi_j^`(v_j(n))\f$.
 * 
 * @param layer The layer for which we are to calculate the errors.
**/
void NeuralNet::calculateLocalGradients(GenericLayer* layer) {
	// the case for the output layer is simply the disired-actual values.
	if ( layer->childLayer == NULL ) {
		#pragma omp parallel
		for (int i=0; i<layer->numNeurons; ++i) {
			layer->neurons[i].error = this->desiredOutput[i] - layer->neurons[i].value;
		}
	}
	// the case for the hidden layer is the back propogation algorithm
	else if ( layer->parentLayer != NULL ) {
		
	}
}
/**
 * @file GenericLayer.cpp
 * @author Ben Snider
 * @version 0.1
 * 
 * Defines the GenericLayer class for the implementation of the network.
**/ 


#include "GenericLayer.h"
#include <omp.h>
#include <cstdlib>
#include <math.h>

/**
 * Defines the constructor for the generic neural layer.
 * @param numNeurons The number of neurons to create in this layer.
 * @param parent A pointer to the parent layer of this current layer. NULL if this layer is the inputer layer.
 * @param child A pointer to the child layer of this current layer. NULL if this layer is the output layer.
 * @include GenericLayer-constructor-example.cpp
 * @throw integer Throws an integer exception when the bounds of the GenericLayer::MAX_NEURONS or GenericLayer::MIN_NEURONS number of neurons in this layer has been broken. 
 * @post All the default values have been assigned and all arrays have been allocated.
**/ 
GenericLayer::GenericLayer(int numNeurons, GenericLayer* parent, GenericLayer* child) {
	if ( numNeurons >= this->MIN_NEURONS && numNeurons <= this->MAX_NEURONS ) {
		this->numNeurons = numNeurons;
	}
	else {
		throw numNeurons;
	}
	this->parentLayer = parent;
	this->childLayer = child;
	
	this->initWeights();
	this->initNeurons();
}

/**
 * Since the layers are created in a linked manner, such that the input layer
 * is linked to the hidden layer, etc., we need another method, outside of the
 * constructor so we can create the data structures when all the information
 * is made available. Doing so in the constructor would fail, for example, since creation
 * of the input layer's data structures implies knowledge of the hidden layer's
 * data, which may or may not have been previously defined. In this way we can
 * safegaurd that the data is available.
 * 
 * @pre All layers have been instanciated with a number of neurons to create, a parent layer if any, and a child layer if any.
 * @post The current layer's data structures will all have been created based on the other related layers.
**/
void GenericLayer::init() {
	this->initWeights();
	this->initNeurons();
}

/**
 * Initialize the weights array, but only if we are in a layer with a child because
 * they are responsible for keeping track of the weights.
 * The number of total entries should number GenericLayer.numNeurons*GenericLayer.childLayer.numNeurons.
 * The weights are all initialized to zero before training.
 * @post The array containing the weights for each neuron is allocated and has been filled with zeros. It is ready to be trained.
**/ 
void GenericLayer::initWeights() {
	if ( this->childLayer != NULL ) {
		this->weights = new double*[this->numNeurons];
		
		#pragma omp parallel
		for (int i=0; i<this->numNeurons; ++i) {
			this->weights[i] = new double[this->childLayer->numNeurons];
			for (int j=0; j<this->childLayer->numNeurons; ++i) {
				this->weights[i][j] = 0.0;
			}
		}
	}
}

/**
 * @pre The neuron array is undefined.
 * @post The neuron array has been allocated and intelligent defaults have been set. There is now GenericLayer::numNeurons neurons in this layer. It is ready to be trained.
**/
void GenericLayer::initNeurons() {
	this->neurons = new Neuron[this->numNeurons];
	
	//#pragma omp parallel
	//for (int i=0; i<this->numNeurons; ++i) {
	//	this->neurons[i] = new Neuron();
	//}
}

/** Deallocates the memory from the neuron and weight arrays. **/
GenericLayer::~GenericLayer() {
#pragma omp parallel
	for (int i=0; i<this->numNeurons; ++i) {
		delete[] this->weights[i];
	}
	delete[] this->weights;
	delete[] this->neurons;
}



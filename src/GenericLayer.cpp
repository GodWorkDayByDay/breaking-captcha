/**
 * @file GenericLayer.cpp
 * @author Ben Snider
 * @version 0.1
 * 
 * Defines the GenericLayer class for the implementation of the network.
**/ 

#include "GenericLayer.h"

GenericLayer::GenericLayer(int numNeurons, const GenericLayer& parent, const GenericLayer& child) throw(int) {
	if ( numNeurons >= this->MIN_NEURONS && numNeurons <= this->MAX_NEURONS ) {
		this->numNeurons = numNeurons;
	}
	else {
		throw numNeurons;
	}
	this->parentLayer = parent;
	this->childLayer = child;
	this->hasParent = true;
	this->hasChild = true;
}

GenericLayer::GenericLayer(const GenericLayer& parent, const GenericLayer& child) {
	this->numNeurons = 0;
	this->parentLayer = parent;
	this->childLayer = child;
	this->hasParent = true;
	this->hasChild = true;
}

GenericLayer::GenericLayer(int numNeurons) {
	if ( numNeurons >= this->MIN_NEURONS && numNeurons <= this->MAX_NEURONS ) {
		this->numNeurons = numNeurons;
	}
	else {
		throw numNeurons;
	}
	
	this->hasParent = false;
	this->hasChild = false;
}

GenericLayer::GenericLayer() {
	this->numNeurons = 0;
	this->hasParent = false;
	this->hasChild = false;
}

void GenericLayer::init() {
	this->initWeights();
	this->initNeurons();
}
 
void GenericLayer::initWeights() {
	if ( this->hasChild && this->numNeurons != 0 ) {
		#pragma omp parallel
		for (int i=0; i<this->numNeurons; ++i) {
			for (int j=0; j<this->childLayer.numNeurons; ++i) {
				this->weights.at(i).at(j) = 0.0;
			}
		}
	}
}

void GenericLayer::initNeurons() {
	#pragma omp parallel
	for (int i=0; i<this->numNeurons; ++i) {
		this->neurons.push_back(0.0);
	}
}

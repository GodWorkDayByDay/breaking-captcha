/**
 * @file GenericLayer.cpp
 * @author Ben Snider
 * @version 0.1
 * 
 * Defines the GenericLayer class for the implementation of the network.
**/ 

#include "GenericLayer.h"

GenericLayer::GenericLayer(int numNeurons, GenericLayer& parent, GenericLayer& child) throw(int) {
	if ( numNeurons >= this->MIN_NEURONS && numNeurons <= this->MAX_NEURONS ) {
		this->numNeurons = numNeurons;
	}
	else {
		throw numNeurons;
	}
	this->parentLayer = &parent;
	this->childLayer = &child;
	this->hasParent = true;
	this->hasChild = true;
}

GenericLayer::GenericLayer(GenericLayer& parent, GenericLayer& child) {
	this->numNeurons = 0;
	this->parentLayer = &parent;
	this->childLayer = &child;
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

void GenericLayer::setParent(GenericLayer& parent) {
	this->parentLayer = &parent;
	this->hasParent = true;
}

void GenericLayer::setChild(GenericLayer& child) {
	this->childLayer = &child;
	this->hasChild = true;
}

void GenericLayer::setNumNeurons(int numNeurons) {
	this->numNeurons = numNeurons;
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
	int j;
	
	if ( this->hasChild and this->numNeurons > 0 and this->childLayer->numNeurons > 0) {
		this->weights.clear();
		this->weights.resize(this->numNeurons);
		
//		#pragma omp parallel for private(j) if(this->numNeurons > 500)
		for (int i=0; i<this->numNeurons; ++i) {
			this->weights.at(i).clear();
			this->weights.at(i).resize(this->childLayer->numNeurons);
			
			for (j=0; j<this->childLayer->numNeurons; ++j) {
				this->weights.at(i).at(j) = 0.0;
			}
		}
	}
}

void GenericLayer::initNeurons() {
	this->neurons.clear();
	this->neurons.resize(numNeurons);
	
//	#pragma omp parallel for if(this->numNeurons > 500)
	for (int i=0; i<this->numNeurons; ++i) {
		this->neurons.at(i).value = 0.0;
	}
}

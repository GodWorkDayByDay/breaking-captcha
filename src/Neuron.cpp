/**
 * @file Neuron.cpp
 * @author Ben Snider
 * @version 0.1
 * 
 * Implements the Neuron class.
**/

#include "Neuron.h"

/**
 * Defines the constructor for the Neuron class.
**/
Neuron::Neuron() {
	this->value = 0;
	this->bias = 1;
	this->biasWeight = 0;
	this->error = 0.0;
	this->localGradient = 0;
}

/** Neuron class desctructor. **/
Neuron::~Neuron() { }

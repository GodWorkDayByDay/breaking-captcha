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
 * @param value A double representing the value of the neuron.
 * @param bias A double representing the bias of the neuron.
 * @param biasWeight A double representing the weight of the bias on the given neuron.
**/
Neuron::Neuron(double value=0.0, double bias=1.0, double biasWeight=0.0) {
	this.value = value;
	this.bias = bias;
	this.biasWeight = biasWeight;
	this.error = 0.0;
}

/** Neuron class desctructor. **/
Neuron::~Neuron() {
}

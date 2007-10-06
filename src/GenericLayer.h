/**
 * @file GenericLayer.h
 * @author Ben Snider
 * @version 0.1
 * 
 * Header file for the GenericLayer class.
**/ 

#include "Neuron.h"

#ifndef GENERICLAYER_H_
#define GENERICLAYER_H_

/**
 * @class GenericLayer
 * 
 * Provides a generic implementation of a neural network layer.
 * In general, the network is defined as loosely coupled neurons in a feed
 * forward three layered design. The neurons are all each connected to adjacent layer neurons,
 * and those connections have weights associated with them. In this implementation,
 * those neurons with a child layer will be associated with the weights between
 * them and their children. Also included are bias factors, which are stored in
 * the neurons themselves since there is only one bias per neuron. The neurons
 * will be initialized with values of 0 except for the bias value which will be 1.
 * The error factor is also included on the neuron itself.
**/ 
class GenericLayer {
private:
	static const int MAX_NEURONS = 1000; /**< Maximum number of neurons allowed in this layer. **/
	static const int MIN_NEURONS = 2; /**< Minimum number of neurons allowed in this layer **/
public:
	GenericLayer(int numNeurons, GenericLayer* parent, GenericLayer* child);
	virtual ~GenericLayer();
	void initWeights();
	void initNeurons();
	GenericLayer* parentLayer; /**< Pointer to the parent layer of this layer. **/
	GenericLayer* childLayer; /**< Pointer to the child layer of this layer. **/
	Neuron* neurons; /**< The array of neurons that populate this layer of the network. **/
	int numNeurons; /**< The number of neurons to create in this layer. **/
	double** weights; /**< The 2D array of weights with dimensions of this layer's numNeurons by the child layer's numNeruons. If there is no child layer then it remains NULL. **/
};

#endif /*GENERICLAYER_H_*/

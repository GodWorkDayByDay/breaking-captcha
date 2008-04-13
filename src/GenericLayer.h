/**
 * @file GenericLayer.h
 * @author Ben Snider
 * @version 0.1
 * 
 * Header file for the GenericLayer class.
**/ 

#include <vector>
#include <omp.h>
#include <cstdlib>
#include <math.h>
#include "MersenneTwister.h"
#include "Random.h"
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
	static const int MAX_NEURONS = 10000; /**< Maximum number of neurons allowed in this layer. **/
	static const int MIN_NEURONS = 2; /**< Minimum number of neurons allowed in this layer **/
public:
	/**
	 * Create a neural layer with numNeurons, parent and child.
	 * @param numNeurons The number of neurons to create in this layer.
	 * @param parent The parent layer of this current layer.
	 * @param child The child layer of this current layer.
	 * @include GenericLayer-constructor-example.cpp
	 * @throw integer Throws an integer exception when the bounds of the GenericLayer::MAX_NEURONS or GenericLayer::MIN_NEURONS number of neurons in this layer has been broken. 
	 * @post All the default values have been assigned and all vectors have been filled. Ready to train.
	**/ 
	GenericLayer(int numNeurons, GenericLayer& parent, GenericLayer& child) throw(int);
	
	/**
	 * Create a neural layer with parent and child.
	 * @param parent The parent layer of this layer.
	 * @param child The child layer of this layer
	 * @poast Parent and child layers have been assigned, but still need to
	 * assign the number of neurons in this layer.
	**/
	GenericLayer(GenericLayer& parent, GenericLayer& child);
	
	/**
	 * Create a neual layer with numNeruons.
	 * @param numNeruons The number of neurons in this layer.
	 * @post The number of neurons for this layer has been assigned but still
	 * need either a parent or child layer, or both, to function.
	**/
	GenericLayer(int numNeurons);
	
	/**
	 * Create a default neural layer.
	 * @post Defaults have been assigned to all member variables.
	**/
	GenericLayer();
	
	/**
	 * Set the parent layer of this layer.
	 * @param parent The parent layer to assign to this layer.
	**/
	void setParent(GenericLayer& parent);
	
	/**
	 * Set the child layer of this layer.
	 * @param child The child layer to assign to this layer.
	**/
	void setChild(GenericLayer& child);
	
	/**
	 * Set the number of neurons to create for this layer.
	 * @param numNeurons The number of neurons to create for this layer.
	**/
	void setNumNeurons(int numNeurons);
	
	/**
	 * Initialize the weights array, but only if we are in a layer with a child because
	 * they are responsible for keeping track of the weights.
	 * The number of total entries should number GenericLayer.numNeurons*GenericLayer.childLayer.numNeurons.
	 * The weights are all initialized to zero before training.
	 * @post The array containing the weights for each neuron is allocated and has been filled with zeros. It is ready to be trained.
	**/
	void initWeights();
	
	/**
	 * Initialize the starting values of all the neurons.
	 * @pre The neuron array is undefined.
	 * @post The neuron array has been allocated and intelligent defaults have been set. There is now GenericLayer::numNeurons neurons in this layer. It is ready to be trained.
	**/
	void initNeurons();
	
	/**
	 * Since the layers are created in a linked manner, such that the input layer
	 * is linked to the hidden layer, etc., we need another method, outside of the
	 * constructor so we can create the data structures when all the information
	 * is made available. Doing so in the constructor would fail, for example, since creation
	 * of the input layer's data structures implies knowledge of the hidden layer's
	 * data, which may or may not have been previously defined. In this way we can
	 * safegaurd that the data is available.
	 * 
	 * @pre All layers have been instantiated with the number of neurons to create, a parent layer if any, and a child layer if any.
	 * @post The current layer's data structures will all have been created based on the other related layers.
	**/
	void init();
	
	GenericLayer* parentLayer; /**< Pointer to the parent layer of this layer. **/
	GenericLayer* childLayer; /**< Pointer to the child layer of this layer. **/
	
	bool hasParent; /**< Does this layer have a parent layer? **/
	bool hasChild; /**< Does this layer have a child layer? **/
	
	int numNeurons; /**< The number of neurons to create in this layer. **/
	std::vector<Neuron> neurons; /**< The array of neurons that populate this layer of the network. **/
	std::vector<std::vector<double> > weights; /**< The 2D array of weights with dimensions of this layer's numNeurons by the child layer's numNeruons. If there is no child layer then it remains NULL. **/
};

#endif /*GENERICLAYER_H_*/

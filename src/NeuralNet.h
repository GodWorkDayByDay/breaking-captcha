/**
 * @file NeuralNet.h
 * @author Ben Snider
 * @version 0.1
 * 
 * Defines the NeuralNet class.
**/

#ifndef NEURALNET_H_
#define NEURALNET_H_

#include "GenericLayer.h"

/**
 * @class NeuralNet
 * This class brings together the work done by various parts of the Neural system.
 * It is the binding layer between the neural layers and the data structures
 * inherent in the system. As such, it is responsible for initializing,
 * stucturing, training, and calculating the neural network environment.
**/
class NeuralNet
{
public:
	NeuralNet(int numInput, int numHidden, int numOutput);
	void train();
	void compute();
	void calculateNeuronValues(GenericLayer* layer);
	double logisticActivation(double x);
	void alterWeights(GenericLayer* layer);
	double calculateMSE();
	Neuron* getOutput();
	
	double learningRate; /**< This variable controls the rate at which the network learns. It is responsible for smoothing out the learning functions. **/
	int maxTrainingIterations; /**< The max number of iterations to compute while training. **/
	double percentChange; /**< When to stop the training based on each epoch's mean squared error. A percentage of the rate of change. **/
	
	GenericLayer* input; /**< The input layer to the neural network. **/
	GenericLayer* hidden; /**< The hidden layer to the neural network. **/
	GenericLayer* output; /**< The output layer to the neural network. **/
	
	int numInput; /**< The number of input neurons to create. **/
	int numHidden; /**< The number of hidden neurons to create. **/
	int numOutput; /**< The number of output neurons to create. **/
	
	double* desiredOutput; /**< The expected results from the training data. Each element is related to each output neuron's expected value **/
	double* inputData; /**< The data to be calculated from the environment, in the same form as each set of data in trainingData. **/
};

#endif /*NEURALNET_H_*/

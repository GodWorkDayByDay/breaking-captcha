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
	
	GenericLayer* input; /**< The input layer to the neural network. **/
	GenericLayer* hidden; /**< The hidden layer to the neural network. **/
	GenericLayer* output; /**< The output layer to the neural network. **/
	
	int numInput; /**< The number of input neurons to create. **/
	int numHidden; /**< The number of hidden neurons to create. **/
	int numOutput; /**< The number of output neurons to create. **/
	
	double** trainingInput; /**< The data, in a 2D array, that the network will be trained against. Each index will be a set of data, and each sub-index will be a data point such that trainingData[i][j] represents trainging sample i and datum j. **/
	double** desiredValues; /**< The expected results from the training data. Each (i,j) element is related to each neuron's expected value, such that index (i) is the set of expected values for the entire set of neurons based on the calculations done from trainingData[i]. **/
	double* inputData; /**< The data to be calculated from the environment, in the same form as each set of data in trainingData. **/
};

#endif /*NEURALNET_H_*/

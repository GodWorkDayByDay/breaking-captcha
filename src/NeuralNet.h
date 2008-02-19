/**
 * @file NeuralNet.h
 * @author Ben Snider
 * @version 0.1
 * 
 * Defines the NeuralNet class.
**/

#ifndef NEURALNET_H_
#define NEURALNET_H_

#include <cstdlib>
#include <cmath>
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
	/**
	 * Constructs a new neural network based on the number of neurons in each layer.
	 * @param numInput The number of neurons to create in the input layer.
	 * @param numHidden The number of neurons to create in the hidden layer.
	 * @param numOutput The number of neurons to create in the output layer.
	**/
	NeuralNet(int numInput, int numHidden, int numOutput);
	
	/**
	 * Default constructor makes an empty NN.
	**/
	NeuralNet();
	
	/**
	 * Trains the NN according to the values in inputData and desiredOutput.
	 * @pre The trainingData and desiredValues arrays have been assigned and are of the appropriate dimensions for this particular NN.
	 * @post All the weights and biases are set according to the trainingData and the NN is ready for real environment data. 
	**/
	void train();
	
	/**
	 * Computes the output layer from the inputData.
	 * @pre The NN has been initialized and trained, and the inputData array has been assigned and is of the appropriate dimensions for this particular NN.
	 * @post The output layer nodes now contain, based on the NN's weights and biases, the output values for the given data set.
	**/
	void compute();
	
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
	 * @param layer The layer to calculate the values of the neurons.
	**/
	void calculateNeuronValues(GenericLayer& layer);
	
	/**
	 * Used for computing the value of a neuron after calculating the raw value.
	 * We use an easily differentiable function so that it is easier later on to calculate the
	 * change in weights. The particular function we use here is the sigmoidal activation
	 * function, which constrains the output to between 0 and +1, a good fit for NNs.
	 * 
	 * The precise function we use is \f$\displaystyle\phi_j(x) = \frac{1}{1 + exp(-x)}\f$.
	 * @param x The variable to evaulate the function at, ie f(x).
	**/
	double logisticActivation(double x);
	
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
	void alterWeights(GenericLayer& layer);
	
	/**
	 * Calculates the mean squared error of the output layer from the desiredOutput
	 * @return The mean squared error of the output layer.
	**/
	double calculateMSE();
	
	/**
	 * Returns the output layer of this NN.
	 * @return The array of output neurons resulting from a previous calculation.
	**/
	std::vector<Neuron> getOutput();
	
	double learningRate; /**< This variable controls the rate at which the network learns. It is responsible for smoothing out the learning functions. **/
	int maxTrainingIterations; /**< The max number of iterations to compute while training. **/
	double percentChange; /**< When to stop the training based on each epoch's mean squared error. A percentage of the rate of change. **/
	
	GenericLayer input; /**< The input layer to the neural network. **/
	GenericLayer hidden; /**< The hidden layer to the neural network. **/
	GenericLayer output; /**< The output layer to the neural network. **/
	
	int numInput; /**< The number of input neurons to create. **/
	int numHidden; /**< The number of hidden neurons to create. **/
	int numOutput; /**< The number of output neurons to create. **/
	
	std::vector<double> desiredOutput; /**< The expected results from the training data. Each element is related to each output neuron's expected value **/
	std::vector<double> inputData; /**< The data to be calculated from the environment, in the same form as each set of data in trainingData. **/
};

#endif /*NEURALNET_H_*/

/**
 * @file Neuron.h
 * @author Ben Snider
 * @version 0.1
 * 
 * Defines the Neuron class.
**/

#ifndef NEURON_H_
#define NEURON_H_

/**
 * @class Neuron
 * 
 * Provides the structure of the neuron.
 * The Neuron class has details about the neuron's value, which is the value
 * stored by the neruon. The neuron's bias, which is used and adjusted when training and
 * computing new values for the network. The neuron bias' weight which is used similarly
 * to the bias. And also the error, which is used primarily in training for computing how
 * far off the network's output is from the desired output. All of this is used generically
 * and in conjuction with the GenericLayer class to implement any sort of nerual network
 * layout.
**/
class Neuron {
public:
	Neuron();
	//Neuron(double value, double bias, double biasWeight);
	double value; /**< The value of the neuron. **/
	double bias; /**< The bias of the neuron. **/
	double biasWeight; /**< The weight of the bias on the neuron. **/
	double error; /**< The error associated with the computation of the network versus the desired output. **/
	double localGradient; /**< The local gradient associated with finding delta values for weight changes. **/
};

#endif /*NEURON_H_*/

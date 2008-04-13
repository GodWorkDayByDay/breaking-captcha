#include "../src/NeuralNet.h"
#include "boost/test/minimal.hpp"
#include <math.h>
#include <iostream>
using namespace std;

#define PRINT_TOKEN(token) printf(#token " = %f\n", token)

int test_main( int, char *[] ) {
	double limit = 0.000001;
	NeuralNet nn(2,2,1);
	
	// test calculateMSE
	nn.output.neurons.at(0).error = 1;
	nn.desiredOutput.clear();
	nn.desiredOutput.push_back(0);
	BOOST_CHECK( nn.calculateMSE() == 1 );
	
	// test calculateNeuronValues for input layer
	nn.inputData.clear();
	nn.inputData.push_back(9);
	nn.inputData.push_back(104);
	nn.calculateNeuronValues(nn.input);
	BOOST_CHECK( nn.input.neurons.at(0).value == 9 );
	BOOST_CHECK( nn.input.neurons.at(1).value == 104 );
	
	// test calculateNeuronValues for hidden layer
	nn.input.neurons.at(0).value = 5;
	nn.input.neurons.at(1).value = 7;
	nn.input.weights.at(0).at(0) = 2;
	nn.input.weights.at(0).at(1) = 2;
	nn.input.weights.at(1).at(0) = 3;
	nn.input.weights.at(1).at(1) = 3;
	nn.calculateNeuronValues(nn.hidden);
	BOOST_CHECK( fabs(nn.hidden.neurons.at(0).value - nn.logisticActivation(20)) < limit ); 
	BOOST_CHECK( fabs(nn.hidden.neurons.at(1).value - nn.logisticActivation(42)) < limit );
//	
//	// test the XOR problem against just calculateNeuronValues,
//	// variate the weights manually to exclude alterWeights
//	nn.input.weights.at(0).at(0) = 1;
//	nn.input.weights.at(0).at(1) = -1;
//	nn.input.weights.at(1).at(0) = -1;
//	nn.input.weights.at(1).at(1) = 1;
//	
//	nn.inputData.clear();
//	nn.inputData.push_back(1);
//	nn.inputData.push_back(0);
//	nn.calculateNeuronValues(nn.input);
//	nn.calculateNeuronValues(nn.hidden);
//	nn.calculateNeuronValues(nn.output);
//	BOOST_CHECK( fabs(nn.output.neurons.at(0).value - 1) < limit );
//	
//	nn.inputData.clear();
//	nn.inputData.push_back(0);
//	nn.inputData.push_back(0);
//	nn.calculateNeuronValues(nn.input);
//	nn.calculateNeuronValues(nn.hidden);
//	nn.calculateNeuronValues(nn.output);
//	BOOST_CHECK( fabs(nn.output.neurons.at(0).value - 0) < limit );
	
	// 2 input, 2 middle, 1 output
	NeuralNet nn1 = NeuralNet(2,2,1);
	nn1.maxTrainingIterations = 500000000;
	nn1.learningRate = 0.01;
	
	// need 4 cases, each with 2 input and 1 output
	nn1.trainingInput.resize(4);
	nn1.trainingInput.at(0).resize(2);
	nn1.trainingInput.at(1).resize(2);
	nn1.trainingInput.at(2).resize(2);
	nn1.trainingInput.at(3).resize(2);
	nn1.trainingOutput.resize(4);
	nn1.trainingOutput.at(0).resize(1);
	nn1.trainingOutput.at(1).resize(1);
	nn1.trainingOutput.at(2).resize(1);
	nn1.trainingOutput.at(3).resize(1);
	
	// 0 XOR 0 = 0
	nn1.trainingInput.at(0).at(0) = 0;
	nn1.trainingInput.at(0).at(1) = 0;
	nn1.trainingOutput.at(0).at(0) = 0;
	
	// 0 XOR 1 = 1
	nn1.trainingInput.at(2).at(0) = 0;
	nn1.trainingInput.at(2).at(1) = 1;
	nn1.trainingOutput.at(2).at(0) = 1;
	
	// 1 XOR 0 = 1
	nn1.trainingInput.at(3).at(0) = 1;
	nn1.trainingInput.at(3).at(1) = 0;
	nn1.trainingOutput.at(3).at(0) = 1;
	
	// 1 XOR 1 = 0
	nn1.trainingInput.at(1).at(0) = 1;
	nn1.trainingInput.at(1).at(1) = 1;
	nn1.trainingOutput.at(1).at(0) = 0;
	
	nn1.train();
	
	// test in a binary order
	nn1.inputData.clear();
	nn1.inputData.push_back(0);
	nn1.inputData.push_back(0);
	nn1.compute();
	cout << "0 XOR 0 = " << nn1.output.neurons.at(0).value << endl;
	BOOST_CHECK( nn1.output.neurons.at(0).value == 0 );
	
	nn1.inputData.clear();
	nn1.inputData.push_back(0);
	nn1.inputData.push_back(1);
	nn1.compute();
	cout << "0 XOR 1 = " << nn1.output.neurons.at(0).value << endl;
	BOOST_CHECK( nn1.output.neurons.at(0).value == 1 );
	
	nn1.inputData.clear();
	nn1.inputData.push_back(1);
	nn1.inputData.push_back(0);
	nn1.compute();
	cout << "1 XOR 0 = " << nn1.output.neurons.at(0).value << endl;
	BOOST_CHECK( nn1.output.neurons.at(0).value == 1 );
	
	nn1.inputData.clear();
	nn1.inputData.push_back(1);
	nn1.inputData.push_back(1);
	nn1.compute();
	cout << "1 XOR 1 = " << nn1.output.neurons.at(0).value << endl;
	BOOST_CHECK( nn1.output.neurons.at(0).value == 0 );
	
	return 0;
}

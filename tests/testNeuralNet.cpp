#include "../src/NeuralNet.h"
//#include "boost/test/minimal.hpp"
#include <iostream>
using namespace std;

int main( int, char *[] ) {
	NeuralNet nn(2,2,1);
	
	// 0 XOR 0 = 0
	nn.inputData.clear();
	nn.inputData.push_back(0);
	nn.inputData.push_back(0);
	nn.desiredOutput.clear();
	nn.desiredOutput.push_back(0);
	nn.train();
	
	// 1 XOR 1 = 0
	nn.inputData.clear();
	nn.inputData.push_back(1);
	nn.inputData.push_back(1);
	nn.desiredOutput.clear();
	nn.desiredOutput.push_back(0);
	nn.train();
	
	// 0 XOR 1 = 1
	nn.inputData.clear();
	nn.inputData.push_back(0);
	nn.inputData.push_back(1);
	nn.desiredOutput.clear();
	nn.desiredOutput.push_back(1);
	nn.train();
	
	// 1 XOR 0 = 1
	nn.inputData.clear();
	nn.inputData.push_back(1);
	nn.inputData.push_back(0);
	nn.desiredOutput.clear();
	nn.desiredOutput.push_back(1);
	nn.train();
	
	// test in a binary order
	nn.inputData.clear();
	nn.inputData.push_back(0);
	nn.inputData.push_back(0);
	nn.compute();
	if ( nn.output.neurons.at(0).value == 0 ) cout << "00 works\n";
	
	nn.inputData.clear();
	nn.inputData.push_back(0);
	nn.inputData.push_back(1);
	nn.compute();
	if ( nn.output.neurons.at(0).value == 1 ) cout << "01 works\n";
	
	nn.inputData.clear();
	nn.inputData.push_back(1);
	nn.inputData.push_back(0);
	nn.compute();
	if ( nn.output.neurons.at(0).value == 1 ) cout << "10 works\n";
	
	nn.inputData.clear();
	nn.inputData.push_back(1);
	nn.inputData.push_back(1);
	nn.compute();
	if ( nn.output.neurons.at(0).value == 0 ) cout << "11 works\n";
	
	return 0;
}

#ifndef NNTRAINER_H_
#define NNTRAINER_H_

#include <string>
#include <list>
#include "NeuralNet.h"

/**
 * This class is meant to be used in conjunction with the NeuralNet class to
 * provide an easy interface to train a network. It requires a directory of
 * training images, named to indicate which character they depict. To
 * allow multiple images of the same character the images can be numbered. So
 * for example the directory could contain the following:
 * 	s.gif g.gif g1.gif g2.gif G.gif 3.gif ...
*/
class NNTrainer {
public:
	/**
	 * Constructs a new trainer using nnet and location.
	**/
	NNTrainer(NeuralNet& nnet, std::string& location): nn(nnet), location(imgLocation), imgToChars() {}
	
	/**
	 * Constructs the default trainer, must later set nn and imgLocation.
	**/
	NNTrainer(): nn(), imgLocation(), imgToChars() {}
	
	/**
	 * Does the actual training of the NN, so that it is ready to compute.
	**/
	void train();
	
	/** Setter method to set the network to train. **/
	void setNN(NeuralNet& nnet): nn(nnet) {}
	
	/** 
	 * Setter method to set the location of the training images. Causes a call
	 * to readFiles to update imgToChars.
	**/
	void setLocation(std::string& loc): imgLocation(loc) { this->readFiles(); }
	
private:
	/**
	 * Creates a list of pairs of image locations to the characters they depict.
	 * Can't use a map here since we sometimes want multiple images depicting
	 * the same characters.
	**/
	void readFiles();
	
	std::list<std::pair<std::string, std::string> > imgToChars; /**< Maps the training image locations to which character they depict. **/
	std::string imgLocation; /**< The directory where all the training images can be found. **/
	NeralNet nn; /**< The neural network this class is to train. **/
};

#endif /*NNTRAINER_H_*/

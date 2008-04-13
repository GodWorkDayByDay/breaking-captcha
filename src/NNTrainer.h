#ifndef NNTRAINER_H_
#define NNTRAINER_H_

#include <string>
#include <vector>
#include <map>
#include <omp.h>
#include "Magick++.h"
#include "boost/filesystem.hpp"
#include "NeuralNet.h"

namespace fs = boost::filesystem;
typedef boost::filesystem::path path_t;
typedef std::string char_t;
typedef std::map<char_t, char_t> img_char_map_t;

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
	NNTrainer(NeuralNet& nnet, std::string& loc):
		nn(nnet), imgPath(fs::path(loc)), imgToChars() {}
	
	/**
	 * Constructs the default trainer, must later set nn and imgLocation.
	**/
	NNTrainer(): nn(), imgPath(), imgToChars() {}
	
	/**
	 * Does the actual training of the NN, so that it is ready to compute.
	**/
	void train();
	
	/** Setter method to set the network to train. **/
	void setNN(NeuralNet& nnet) { this->nn = nnet; }
	
	/** 
	 * Setter method to set the location of the training images.
	**/
	void setLocation(std::string& loc);
	
//private:
	/**
	 * Creates a vector of pairs of image locations to the characters they depict.
	**/
	void readFiles();
	
	/**
	 * Read the pixels from imgFileName and store them in imgPixels.
	 * @param imgFileName The string that represents an image file on a storage medium.
	 * @param imgPixels The vector to contain the pixel values of the image represented by imgFileName.
	**/
	void readPixels(char_t imgFileName, std::vector<double>& imgPixels);
	
	/**
	 * Populates a vector with the filenames found in imgLocation. Only stores
	 * the fs::path representation.
	 * \param names The vector of paths to populate with filenames.
	**/
	void getFileNames(std::vector<path_t>& names);
	
	NeuralNet nn; /**< The neural network this class is to train. **/
	path_t imgPath; /**< The directory where all the training images can be found. **/
	img_char_map_t imgToChars; /**< Maps the training image locations to which character they depict. **/
};

#endif /*NNTRAINER_H_*/

#include "NNTrainer.h"

void NNTrainer::train() {
	this->readFiles();
}

void NNTrainer::readFiles() {
	std::vector<std::string> filenames;
	std::string character;
	// TODO: put names of file in directory imgLocation into filenames;
	#pragma omp parallel for
	for (int i=0; i<filenames.size(); ++i) {
		character = filenames.at(i).find_first_of(".");
		std::pair<std::string, std::string> tmpPair(filenames.at(i), character);
		this->imgToChars.insert(tmpPair);
		
	}
}

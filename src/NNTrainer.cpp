#include "NNTrainer.h"

void NNTrainer::train() {
	this->readFiles();
}

void NNTrainer::setLocation(std::string& loc) {
	this->imgPath = fs::system_complete(fs::path(loc, fs::native));
	this->readFiles();
}

void NNTrainer::getFileNames(std::vector<path_t>& names) {
	names.clear();
	
	if (fs::is_directory(this->imgPath)) {
		fs::directory_iterator end_iter, dir_itr(this->imgPath);
		for ( ; dir_itr != end_iter; ++dir_itr ) {
			if (fs::is_regular(dir_itr->status())) {
				names.push_back(dir_itr->path());
			}
		}
	}
}

void NNTrainer::readFiles() {
	std::vector<path_t> filenames;
	this->getFileNames(filenames);
	char_t character;
	
	#pragma omp parallel for
	for (int i=0; i<filenames.size(); ++i) {
		// only need the first character in the filename, the rest is numbering and extension
		character = filenames.at(i).leaf().at(0);
		
		// we want the full path of the image mapped to the character of the image
		// amounts to less processing later
		std::pair<char_t, char_t> tmpPair(filenames.at(i).native_file_string(), character);
		this->imgToChars.insert(tmpPair);
	}
}

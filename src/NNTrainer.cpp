#include "NNTrainer.h"

void NNTrainer::train() {
	std::vector<double> tmpPixelValues;
	std::string outputCharacters = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
	this->readFiles();
	
	for (img_char_map_t::iterator imgCharItr = this->imgToChars.begin(); imgCharItr != this->imgToChars.end(); ++imgCharItr) {
		this->nn.inputData.clear();
		this->readPixels(imgCharItr->first, tmpPixelValues);
		this->nn.inputData = tmpPixelValues;
		
		this->nn.desiredOutput.assign(this->nn.numOutput, 0.0);
		this->nn.desiredOutput.at(outputCharacters.find(imgCharItr->second)) = 1.0;
		
		this->nn.train();
	}
}

void NNTrainer::readPixels(char_t imgFileName, std::vector<double>& imgPixels) {
	Magick::Image image(imgFileName);
	image.type(Magick::BilevelType);
	imgPixels.clear();
		
	// move the image data into the pixel values structure
	for (unsigned long h=0; h<image.rows(); ++h) {
		for (unsigned long w=0; w<image.columns(); ++w) {
			imgPixels.push_back((double)(((Magick::ColorMono)image.pixelColor(w, h)).mono()));
		}
	}
}

void NNTrainer::setLocation(std::string& loc) {
	this->imgPath = fs::path(loc);
}

void NNTrainer::getFileNames(std::vector<path_t>& names) {
	names.clear();
	
	if (fs::is_directory(this->imgPath)) {
		fs::directory_iterator end_iter, dir_itr(this->imgPath);
		for ( ; dir_itr != end_iter; ++dir_itr ) {
			if (fs::is_regular(dir_itr->status()) && dir_itr->path().leaf().at(0) != '.') {
				names.push_back(dir_itr->path());
			}
		}
	}
}

void NNTrainer::readFiles() {
	std::vector<path_t> filenames;
	this->getFileNames(filenames);
	char_t character;
	
	for (unsigned long i=0; i<filenames.size(); ++i) {
		// only need the first character in the filename, the rest is numbering and extension
		character = filenames.at(i).leaf().at(0);
		
		// we want the full path of the image mapped to the character of the image
		// amounts to less processing later
		std::pair<char_t, char_t> tmpPair(filenames.at(i).native_file_string(), character);
		this->imgToChars.insert(tmpPair);
	}
}

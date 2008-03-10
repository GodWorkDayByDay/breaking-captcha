#include "../src/NNTrainer.h"
#include "boost/test/minimal.hpp"
#include <iostream>
using namespace std;

int test_main( int, char *[] ) {
	NNTrainer trainer;
	vector<path_t> names;
	string designDirString("../design");
	path_t designDir(designDirString);
	path_t homeDir("/home/");
	
	// testing non-default constructor
	{
		NeuralNet n;
		string tmpDirString = "../design";
		trainer = NNTrainer(n, tmpDirString);
	}
	BOOST_CHECK( 0 == trainer.nn.numInput == trainer.nn.numHidden == trainer.nn.numOutput );
	BOOST_CHECK( designDir == trainer.imgPath );
	
	// testing getFileNames().
	vector<path_t> actualHomeNames;
	trainer.imgPath = homeDir;
	trainer.getFileNames(names);
	BOOST_CHECK( actualHomeNames == names );
	
	// another non empty test of getFileNames()
	vector<path_t> actualDesignNames;
	actualDesignNames.push_back(path_t("../design/neural-net.png"));
	actualDesignNames.push_back(path_t("../design/neural-net.class.violet"));
	trainer = NNTrainer();
	trainer.imgPath = designDir;
	trainer.getFileNames(names);
	BOOST_CHECK( actualDesignNames == names );
	
	// testing readFiles()
	img_char_map_t actualMap;
	pair<char_t, char_t> tmpPair;
	tmpPair.first = "../design/neural-net.png";
	tmpPair.second = "n";
	actualMap.insert(tmpPair);
	tmpPair.first = "../design/neural-net.class.violet";
	tmpPair.second = "n"; 
	actualMap.insert(tmpPair);
	trainer = NNTrainer();
	trainer.imgPath = designDir;
	trainer.readFiles();
	BOOST_CHECK( actualMap == trainer.imgToChars );
	
	// testing setLocation()
	trainer = NNTrainer();
	trainer.setLocation(designDirString);
	BOOST_CHECK( designDir == trainer.imgPath );
	
	//TODO: testing readPixels() using a contrived 5x5 b/w image
	
	return 0;
}

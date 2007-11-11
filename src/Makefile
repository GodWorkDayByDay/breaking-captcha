all: GenericLayer.o Neuron.o NeuralNet.o

NeuralNet.o: NeuralNet.cpp NeuralNet.h
	g++ -c -fopenmp NeuralNet.cpp -o NeuralNet.o

GenericLayer.o: GenericLayer.cpp GenericLayer.h
	g++ -c -fopenmp GenericLayer.cpp -o GenericLayer.o

Neuron.o: Neuron.cpp Neuron.h
	g++ -c -fopenmp Neuron.cpp -o Neuron.o

clean:
	rm -fR *.o


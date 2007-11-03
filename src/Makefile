all: GenericLayer.o Neuron.o NeuralNet.o

NeuralNet.o: NeuralNet.cpp NeuralNet.h
	gcc -c -fopenmp NeuralNet.cpp -o NeuralNet.o

GenericLayer.o: GenericLayer.cpp GenericLayer.h
	gcc -c -fopenmp GenericLayer.cpp -o GenericLayer.o

Neuron.o: Neuron.cpp Neuron.h
	gcc -c -fopenmp Neuron.cpp -o Neuron.o

clean:
	rm -fR *.o


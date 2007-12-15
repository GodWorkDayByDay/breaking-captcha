all: Neuron.o GenericLayer.o NeuralNet.o GuessCaptcha.o guess

guess: guess.cpp
	g++ guess.cpp GuessCaptcha.o -o guess

GuessCaptcha.o: GuessCaptcha.cpp
	g++ -c -fopenmp GuessCaptcha.cpp NeuralNet.o -o GuessCaptcha.o

NeuralNet.o: NeuralNet.cpp NeuralNet.h
	g++ -c -fopenmp NeuralNet.cpp GenericLayer.o -o NeuralNet.o

GenericLayer.o: GenericLayer.cpp GenericLayer.h
	g++ -c -fopenmp GenericLayer.cpp Neuron.o -o GenericLayer.o

Neuron.o: Neuron.cpp Neuron.h
	g++ -c -fopenmp Neuron.cpp -o Neuron.o

clean:
	rm -fR *.o


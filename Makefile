CC = g++
C = gcc
ZINC = -I/usr/local/include
INCS = $(ZINC) -I/usr/local/include -I/usr/include
CCFLAGS = -O3 -fopenmp -Wall $(INCS)
CFLAGS = -O3 -Wall $(INCS)
EXECUTABLE = guess
SRCDIR = src

SRC = $(SRCDIR)/Neuron.cpp $(SRCDIR)/GenericLayer.cpp \
      $(SRCDIR)/NeuralNet.cpp $(SRCDIR)/GuessCaptcha.cpp
OBJ = $(SRC:.cpp=.o)
HEADERS = $(SRC:.cpp=.h)

all:: $(SRC) $(EXECUTABLE)

print:
	@echo $(SRC)
	@echo $(OBJ)
	@echo $(HEADERS)

$(OBJ): $(HEADERS)

$(EXECUTABLE): $(OBJ)
	$(CC) $(CCFLAGS) $(OBJ) $(CFLAGS) -o $(EXECUTABLE)

.cpp.o:
	$(CC) -c $(CCFLAGS) $< -o $@

.c.o:
	$(C) -c $(CFLAGS) $< -o $@


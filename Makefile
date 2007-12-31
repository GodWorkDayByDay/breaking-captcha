CC = g++
C = gcc
PNGINC = -I/usr/local/include/libpng12
ZINC = -I/usr/local/include
INCS = $(PNGINC) $(ZINC) -I/usr/local/include -I/usr/include
CCFLAGS = -O3 -fopenmp -Wall $(INCS)
CFLAGS = -O3 -Wall $(INCS)
EXECUTABLE = guess
SRCDIR = src

SRC = $(SRCDIR)/readpng.c $(SRCDIR)/Neuron.cpp $(SRCDIR)/GenericLayer.cpp \
      $(SRCDIR)/NeuralNet.cpp $(SRCDIR)/GuessCaptcha.cpp $(SRCDIR)/guess.cpp
TMP = $(SRC:.cpp=.o)
OBJ = $(TMP:.c=.o)
TMP1 = $(SRC:.cpp=.h)
HEADERS = $(TMP1:.c=.h)

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


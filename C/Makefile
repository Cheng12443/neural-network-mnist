CC = gcc
CFLAGS = -Wall
LIBS = -lm

train_mnist: neural_network.o train_mnist.o
	$(CC) neural_network.o train_mnist.o -o train_mnist $(LIBS)

test_mnist: neural_network.o test_mnist.o
	$(CC) neural_network.o test_mnist.o -o test_mnist $(LIBS)

neural_network.o: neural_network.c neural_network.h
	$(CC) $(CFLAGS) -c neural_network.c

train_mnist.o: train_mnist.c neural_network.h
	$(CC) $(CFLAGS) -c train_mnist.c

test_mnist.o: test_mnist.c neural_network.h
	$(CC) $(CFLAGS) -c test_mnist.c

clean:
	rm -f *.o train_mnist test_mnist 
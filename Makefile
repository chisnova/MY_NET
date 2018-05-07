CC = g++
OBJS = data.o layer.o mnist.o main.o dnn.o
TARGET = my_net
.SUFFIXES: .cc .o

all : $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS)
	
data.o: data.cc data.h
	g++ -c data.cc
layer.o: layer.cc layer.h
	g++ -c layer.cc
dnn.o: dnn.cc dnn.h
	g++ -c dnn.cc
minst.o: mnist.cc mnist.h
	g++ -c minst.cc
nnet_set.o: main.cc
	g++ -c main.cc
clean:
	rm -rf $(OBJS) $(TARGET)

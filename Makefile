CC = g++
OBJS = data.o layer.o mnist.o main.o dnn.o cnn.o w2v.o
TARGET = NNet
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
cnn.o: cnn.cc cnn.h
	g++ -c cnn.cc
w2v.o : w2v.cc w2v.h
	g++ -c w2v.cc
minst.o: mnist.cc mnist.h
	g++ -c minst.cc
main.o: main.cc
	g++ -c main.cc
clean:
	rm -rf $(OBJS) $(TARGET)

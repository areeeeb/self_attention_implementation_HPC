CXX = g++
CXXFLAGS = -std=c++17 -Iinclude -fopenmp -O2
TARGET = main
SRC = main.cpp naive.cpp optimized.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
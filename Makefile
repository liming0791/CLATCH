TARGET_LIB = Clatch.so
DEMO_NAME=clatchdemo
CPP=g++
NVCC=nvcc
ARCH=sm_30
INC=-I/usr/local/cuda/include/ -I/media/ubuntu/Data/liming/opencv3/include
NVCCFLAGS=-Wall -Wextra -Werror -Wshadow -Ofast -fomit-frame-pointer -march=native -funroll-all-loops -fpeel-loops -ftracer -ftree-vectorize -fPIC
CPPFLAGS=-Wall -Wextra -Werror -Wshadow -pedantic -Ofast -std=c++11 -fomit-frame-pointer -march=native -flto -funroll-all-loops -fpeel-loops -ftracer -ftree-vectorize -fPIC
LIBS=-L/usr/local/cuda/lib64 -L/media/ubuntu/Data/liming/opencv3/lib -lcudart -lopencv_core -lopencv_ml -lopencv_cudafeatures2d -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lpthread
CPPSOURCES=$(wildcard *.cpp)
CUSOURCES=$(wildcard *.cu)

OBJECTS=$(CPPSOURCES:.cpp=.o) $(CUSOURCES:.cu=.o)

all: $(CPPSOURCES) $(CUSOURCES)  $(DEMO_NAME) $(TARGET_LIB)

$(TARGET_LIB) : $(OBJECTS) 
	$(CPP) $(CPPFLAGS) $(OBJECTS) -o $@ $(LIBS) -shared

$(DEMO_NAME) : $(OBJECTS) 
	$(CPP) $(CPPFLAGS) $(OBJECTS) -o $@ $(LIBS)

%.o:%.cpp
	$(CPP) -c $(INC) $(CPPFLAGS) $< -o $@

%.o:%.cu
	$(NVCC) --use_fast_math -arch=$(ARCH) -O3 -ccbin $(CC) -std=c++11 -c $(INC) -Xcompiler "$(NVCCFLAGS)" $< -o $@

clean:
	rm -rf *.o $(EXECUTABLE_NAME) $(DEMO_NAME)

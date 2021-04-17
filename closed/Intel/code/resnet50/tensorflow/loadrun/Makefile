LOADGEN_ROOT=/localdisk/amin/workspace/mlperf_inference/loadgen
TF_INSTALL=/localdisk/amin/installations/tf-cc
BOOST_INSTALL=/localdisk/amin/installations/boost-1.7.4
OPENCV_INSTALL=/localdisk/amin/installations/opencv

INCLUDE = -I$(BOOST_INSTALL)/include -I$(LOADGEN_ROOT) -I$(TF_INSTALL)/include -I$(OPENCV_INSTALL)/include/opencv4
LIBOPTS = -L$(BOOST_INSTALL)/lib -L$(LOADGEN_ROOT)/build -L$(TF_INSTALL)/lib -L$(OPENCV_INSTALL)/lib -L$(PWD)/../backend
LDFLAGS := -lpthread -lrt -lmlperf_loadgen -ltensorflow_cc -ltensorflow_backend -lboost_filesystem -lboost_system -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio -lopencv_imgcodecs

CFLAGS = -O3 -fpic -Wall -std=gnu++14 -g
CC = gcc
CXX = g++ -fopenmp
NB_THREADS = 8

.PHONY : all
all : loadrun netrun

loadrun : loadrun.cc inferencer.h multi_instance.h 
	$(CXX) loadrun.cc $(CFLAGS) $(INCLUDE) $(LIBOPTS) -o $@ $(LDFLAGS)

netrun: netrun.cc inferencer.h multi_instance.h 
	$(CXX) netrun.cc $(CFLAGS) $(INCLUDE) $(LIBOPTS) -o $@ $(LDFLAGS)

clean :
	rm -f *.o loadrun netrun

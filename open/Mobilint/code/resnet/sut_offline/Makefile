CXX := g++
CXXFLAGS := -std=c++14 -pthread -O3 -I. -fopenmp -ffast-math
LDFLAGS := -lmaccel -lstdc++fs

OUT_DIR = out

SRCS := $(wildcard *.cc)

all: benchmark debug
	@echo "*** Congratulations! MLPerf benchmark v1.0 compilation is completed. ***"
	@echo "Do not forget to place mlperf.conf and datasets properly!"

benchmark: $(SRCS) lib/libmlperf_loadgen.a
	@mkdir -p out
	$(CXX) $(CXXFLAGS) -o $(OUT_DIR)/$@ $^ $(LDFLAGS)

debug: $(SRCS) lib/libmlperf_loadgen.a
	@mkdir -p out
	$(CXX) $(CXXFLAGS) -o $(OUT_DIR)/benchmark-$@ -g $^ $(LDFLAGS)

clean: $(SUBDIRS_CLEAN)
	@echo "Sweeping all the artifacts..."
	@rm -rf $(OUT_DIR)
PYTHON=python3
NVCC=nvcc
NVCC_FLAGS=\
	-Wno-deprecated-gpu-targets\
	-Xcompiler -Wall\
	-Xcompiler -std=c++14\
	-gencode arch=compute_35,code=compute_35\
	-default-stream per-thread\
	-O3\
	-ftz=true\
	-maxrregcount=32
SHARED_LIBS=-lcublasLt

libmodel.a : model.o
	$(AR) cr $@ model.o

model.o : weights.h model.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ model.cu

test : weights.h example.h model.cu
	$(NVCC) $(NVCC_FLAGS) -DTEST -o $@ model.cu $(SHARED_LIBS)

weights.h : weights.py model.pt
	$(PYTHON) weights.py > $@

example.h : example.py data.gz
	$(PYTHON) example.py > $@

clean :
	$(RM) test libmodel.a model.o weights.h example.h

.PHONY : clean

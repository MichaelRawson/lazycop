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
	-maxrregcount=32\
	--compiler-options -fPIC
SHARED_LIBS=-lcublasLt

libmodel.a : model.o
	$(AR) cr $@ model.o

libmodel.so : model.o
	$(CC) -shared -o $@ $<

model.o : weights.h model.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ model.cu

weights.h : model.pt
	python3 ../model2c.py $< > $@

clean :
	$(RM) test libmodel.a libmodel.so model.o weights.h

.PHONY : clean

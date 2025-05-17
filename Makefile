all: test_cufft libcufft_wrapper.a libcufft_wrapper.so

OPT=-O3
CFLAGS = -g $(OPT)
LIBS=-lcudart -lcuda -lcufft

test_cufft.o: test_cufft.cpp
	g++ -c $< -o $@ $(CFLAGS)

cufft_wrapper.o: cufft_wrapper.cu
	nvcc -c $< -o $@ $(CFLAGS) --cudart=static --cudadevrt=none

test_cufft: test_cufft.o cufft_wrapper.o
	nvcc $^ -o $@ $(CFLAGS) --cudart=static --cudadevrt=none $(LIBS)

libcufft_wrapper.so: cufft_wrapper.o
	g++ --shared -fPIC -o $@ $^ $(LIBS)

libcufft_wrapper.a: cufft_wrapper.o
	ar crv $@ $^
	ranlib $@

clean:
	rm -f *.o

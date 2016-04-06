import arrayfire

def test_fft2(benchmark):
    a = arrayfire.randu(256, 256, dtype=arrayfire.Dtype.c64)
    benchmark(arrayfire.fft2, a)

def test_ifft2(benchmark):
    a = arrayfire.randu(256, 256, dtype=arrayfire.Dtype.c64)
    benchmark(arrayfire.ifft2, a)

# def test_fftshift(benchmark):
#     a = afnumpy.ones((256,256),dtype=numpy.complex64)
#     benchmark(afnumpy.fft.fftshift, a)

# def test_ifftshift(benchmark):
#     a = afnumpy.ones((256,256),dtype=numpy.complex64)
#     benchmark(afnumpy.fft.ifftshift, a)

# def test_array_nocopy(benchmark):
#     a = afnumpy.ones((2048,2048),dtype=numpy.complex64)
#     benchmark(afnumpy.array, a, copy=False)

# def test_array_copy(benchmark):
#     a = afnumpy.ones((2048,2048),dtype=numpy.complex64)
#     benchmark(afnumpy.array, a, copy=True)

# def test_add(benchmark):
#     a = afnumpy.ones((2048,2048),dtype=numpy.complex64)
#     b = afnumpy.ones((2048,2048),dtype=numpy.complex64)
#     benchmark(afnumpy.add, a, b)


        

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_src_path = os.path.dirname(os.path.abspath(__file__))

setup(
    name='bvh_tracing',
    description='CUDA RayTracer with BVH acceleration for 3DGS',
    ext_modules=[
        CUDAExtension(
            name='bvh_tracing._C',
            sources=[os.path.join(_src_path, 'src', f) for f in [
                'bvh.cu',
                'trace.cu',
                'construct.cu',
                'bindings.cpp',
            ]],
            include_dirs=[
                os.path.join(_src_path, 'include'),
            ],
            extra_compile_args={
                "nvcc": ["-O3", "--expt-extended-lambda"],
                "cxx": ["-O3"]}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
)

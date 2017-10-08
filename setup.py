from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Compiler.Options import directive_defaults

#directive_defaults['linetrace'] = True
#directive_defaults['binding'] = True

extensions = [
        Extension('nolips', ['uncurl/nolips.pyx'],
            extra_compile_args=['-O3', '-march=native', '-ffast-math']),
        Extension('sparse_utils', ['uncurl/sparse_utils.pyx'],
            extra_compile_args=['-O3', '-march=native', '-ffast-math'])
        ]

parallel_extensions = [
        Extension('nolips_parallel', ['uncurl/nolips_parallel.pyx'],
            extra_compile_args=['-O3', '-march=native', '-ffast-math', '-fopenmp'],
            extra_link_args=['-fopenmp'])
        ]

try:
    parallel = cythonize(parallel_extensions)
except:
    print('Unable to compile parallel extensions.')
    parallel = []


setup(name='uncurl',
      version='0.2.3',
      description='Tool for clustering single-cell RNASeq data',
      url='https://github.com/yjzhang/uncurl_python',
      author='Yue Zhang',
      author_email='yjzhang@cs.washington.edu',
      license='MIT',
      ext_modules = cythonize(extensions) + parallel,
      packages=find_packages("."),
      install_requires=[
          'numpy',
          'scipy',
          'cython',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)

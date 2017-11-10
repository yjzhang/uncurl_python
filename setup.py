from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize

#directive_defaults['linetrace'] = True
#directive_defaults['binding'] = True

extensions = [
        Extension('uncurl.nolips', ['uncurl/nolips.pyx'],
            extra_compile_args=['-O3', '-march=native', '-ffast-math']),
        Extension('uncurl.sparse_utils', ['uncurl/sparse_utils.pyx'],
            extra_compile_args=['-O3', '-march=native', '-ffast-math'])
        ]

parallel_extensions = [
        Extension('uncurl.nolips_parallel', ['uncurl/nolips_parallel.pyx'],
            extra_compile_args=['-O3', '-march=native', '-ffast-math', '-fopenmp'],
            extra_link_args=['-fopenmp'])
        ]

parallel = []
try:
    parallel = cythonize(parallel_extensions)
except:
    print('Unable to compile parallel extensions.')


setup(name='uncurl',
      version='0.2.4',
      description='Tool for pre-processing single-cell RNASeq data',
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
          'scikit-learn',
      ],
      test_suite='nose.collector',
      tests_require=['nose', 'flaky'],
      zip_safe=False)

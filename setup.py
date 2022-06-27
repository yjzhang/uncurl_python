from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

#directive_defaults['linetrace'] = True
#directive_defaults['binding'] = True

extensions = [
        Extension('uncurl.nolips', ['uncurl/nolips.pyx'],
            extra_compile_args=['-O3', '-ffast-math']),
        Extension('uncurl.sparse_utils', ['uncurl/sparse_utils.pyx'],
            extra_compile_args=['-O3', '-ffast-math'])
        ]

parallel_extensions = [
        Extension('uncurl.nolips_parallel', ['uncurl/nolips_parallel.pyx'],
            extra_compile_args=['-O3', '-ffast-math', '-fopenmp'],
            extra_link_args=['-fopenmp'])
        ]

long_description = ''
with open('README.rst') as f:
    long_description = f.read()

setup(name='uncurl_seq',
      version='0.2.15',
      description='Tool for pre-processing single-cell RNASeq data',
      long_description=long_description,
      long_description_content_type='text/plain',
      url='https://github.com/yjzhang/uncurl_python',
      author='Yue Zhang',
      author_email='yjzhang@cs.washington.edu',
      license='MIT',
      include_dirs=[numpy.get_include()],
      ext_modules = cythonize(extensions + parallel_extensions),
      packages=find_packages("."),
      install_requires=[
          'numpy',
          'scipy',
          'cython',
          'scikit-learn',
      ],
      test_suite='nose.collector',
      tests_require=['nose', 'flaky'],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.5',
      ],
      zip_safe=False)

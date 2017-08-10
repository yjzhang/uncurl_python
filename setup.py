from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Compiler.Options import directive_defaults

directive_defaults['linetrace'] = True
directive_defaults['binding'] = True

extensions = [
        Extension('nolips', ['uncurl/nolips.pyx'], define_macros=[('CYTHON_TRACE', '1')]),
        Extension('sparse_utils', ['uncurl/sparse_utils.pyx'], define_macros=[('CYTHON_TRACE', '1')])
        ]

setup(name='uncurl',
      version='0.2.3',
      description='Tool for clustering single-cell RNASeq data',
      url='https://github.com/yjzhang/uncurl_python',
      author='Yue Zhang',
      author_email='yjzhang@cs.washington.edu',
      license='MIT',
      ext_modules = cythonize(extensions),
      packages=find_packages("."),
      install_requires=[
          'numpy',
          'scipy',
          'cython',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)

from setuptools import setup, find_packages

setup(name='uncurl',
      version='0.2.3',
      description='Tool for clustering single-cell RNASeq data',
      url='https://github.com/yjzhang/uncurl_python',
      author='Yue Zhang',
      author_email='yjzhang@cs.washington.edu',
      license='MIT',
      packages=find_packages("."),
      install_requires=[
          'numpy',
          'scipy',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)

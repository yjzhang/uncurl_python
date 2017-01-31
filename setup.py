from setuptools import setup

setup(name='uncurl',
      version='0.1',
      description='Tool for clustering single-cell RNASeq data',
      url='https://github.com/yjzhang/uncurl_python',
      author='Yue Zhang',
      author_email='yjzhang@cs.washington.edu',
      license='MIT',
      packages=['uncurl'],
      install_requires=[
          'numpy',
      ],
      zip_safe=False)

from setuptools import setup, find_packages

setup(name='detectionmodels',
      version='0.0.1',
      description='Library for models for generating random mouse movements',
      url='',
      author='Joseph Chen',
      author_email='jchen42703@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'tensorflow',
          'pandas',
          'matplotlib'
          'googledrivedownloader'
      ],
      keywords=['deep learning'],
      )

from setuptools import setup, find_packages

setup(name='nspy',
      version='0.1.0',
      description='Python Utilities',
      author='Nitesh Sekhar',
      author_email='niteshsekhar@gmail.com',
      url='https://www.github.com/nitesh1813/',
      packages=find_packages('ns-utils'),
      package_dir={'': 'ns-utils'},
      install_requires=['numpy',  'nibabel'],
     )
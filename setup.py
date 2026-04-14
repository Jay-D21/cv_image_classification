from setuptools import setup, find_packages

setup(
    name='cv_image_classification',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[],
    author='Jay',
    description='A production-ready ML pipeline',
)

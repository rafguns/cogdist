from setuptools import setup
from setuptools.command.test import test as TestCommand
import sys

long_description = open('README.md').read()


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ['--cov=cogdist']
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)


setup(
    name='cogdist',
    version='0.0.1',
    license='New BSD License',
    author='Raf Guns',
    tests_require=['pytest', 'pytest-cov'],
    install_requires=[
        'numpy>=1.17',
        'pandas>=0.25',
    ],
    python_requires='>=3.6',
    cmdclass={'test': PyTest},
    author_email='raf.guns@uantwerpen.be',
    description='Measure cognitive distance between publication portfolios',
    long_description=long_description,
    packages=['cogdist'],
    platforms='any',
    test_suite='tests',
    classifiers=[
        'License :: OSI Approved',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ],
    extras_require={
        'testing': ['pytest'],
    }
)

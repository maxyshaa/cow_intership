
from setuptools import setup

setup(
    name='lm-pkg-Maksimova', version='0.0.1',
    description='Package with linear models for animals value prediction',
    url='https://github.com/maxyshaa/cow_intership.git',
    author='Ksenia Maximova',
    author_email='maxysha95@gmail.com',
    packages= ['blup_animals'],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "pandas>=0.24.2",
        "numpy>=1.16.2"
    ]
)


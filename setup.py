from setuptools import setup, find_packages, Extension
import numpy

setup(
    name='mymodule',
    version='0.1.0',
    author='Yarin&Or',
    description='SymNMF module',
    install_requires=['invoke'],
    packages=find_packages(),
    license='GPL-2',
    ext_modules=[
       Extension("symnmfmodule", sources=["symnmf.c", "symnmfmodule.c"])
    ],
    headers=['symnmf.h']
)
from setuptools import setup, find_packages

setup(
    name='Multi-Task-Learning-with-Limited-Labels',
    version='0.1.0',
    description='Multi-Task-Learning-with-Limited-Labels built upon PyTorch Library for Multi-Task Learning',
    packages=find_packages(),
    install_requires=[
        'torch>=2.1.2',
        'torchvision>=0.16.2',
        'numpy>=1.24.4'
    ],
)

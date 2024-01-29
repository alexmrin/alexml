from setuptools import setup, find_packages

setup(
    name="alexml",
    version="1.0",
    author="Alex Matsumura",
    description="My own high-level PyTorch API",
    packages=find_packages(),
    install_requires=[
        "pytorch",
        "numpy",
        "tqdm",
        "torchvision",
        "matplotlib"
    ],
    entry_points={
        
    }
)
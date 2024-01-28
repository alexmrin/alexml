from setuptools import setup, find_packages

setup(
    name="alexml",
    version="1.0",
    author="Alex Matsumura",
    description="pytorch wrapper",
    packages=find_packages(),
    install_requires=[
        "pytorch",
        "numpy",
        "tqdm"
    ],
    entry_points={
        
    }
)
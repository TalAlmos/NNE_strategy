from setuptools import setup, find_packages

setup(
    name="nne_strategy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "yfinance>=0.2.0"
    ]
)
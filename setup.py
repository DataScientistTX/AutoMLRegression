from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="AutoMLRegression",
    version="0.1.0",
    author="Sercan Gul",
    author_email="your.email@example.com",
    description="An Automated Supervised Machine Learning Regression program",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sercangul/AutoMLRegression",
    packages=find_packages(include=['src', 'src.*']),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.2",
        "pandas>=1.2.4",
        "matplotlib>=3.4.1",
        "scikit-learn>=0.24.2",
        "xgboost>=1.4.1",
        "pyyaml>=5.4.1",
    ],
    entry_points={
        "console_scripts": [
            "auto-ml-regression=src.main:main",
        ],
    },
)
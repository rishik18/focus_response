from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="focus_response",
    version="0.1.0",
    author="Hrishikesh Kanade",
    author_email="rishikanade@outlook.com",
    description="This library provides functionality to measure focus levels in images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/focus_response",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.19.3",
        "opencv-python>=4.5.0",
        "matplotlib>=3.3.0",
        "scipy>=1.5.0",
    ],
)

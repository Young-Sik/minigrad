from setuptools import setup, find_packages

setup(
    name="minigrad",
    version="0.1.0",
    author='Young-Sik Choi',
    author_email="your.email@example.com",
    description="A lightweight autograd engine for educational purposes.",
    organization='Korea Aerospace University',
    url="https://github.com/Young-Sik/minigrad",
    packages=find_packages(include=["minigrad", "minigrad.*"]),
    install_requires=[
        "numpy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)

from setuptools import setup, find_packages

setup(
    name='minigrad',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    author='Young-Sik Choi',
    description='Educational minigrad autograd engine',
    organization='Korea Aerospace University',
    license='MIT',
)

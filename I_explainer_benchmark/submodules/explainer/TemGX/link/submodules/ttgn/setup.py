from setuptools import setup, find_packages

setup(
    name='TTGN',
    version='0.0.1',
    description='An adjusted version of Temporal Graph Nets. Graph Neural Networks for dynamic data. '
                'With support for factual explanations',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'scikit_learn'
    ],
)

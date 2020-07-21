from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="intrinsic-dimensionality",
    version="0.1",
    description='Dense and fastfood transform wrappers to reproduce "Intrinsic dimensionality of objective landscapes" by Li et al. (2018)',
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Jevgenij Gamper",
    author_email="jevgenij.gamper5@gmail.com",
    url="https://github.com/jgamper/intrinsic-dimensionality",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.6"
    ],
)

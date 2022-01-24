import io
from setuptools import find_packages, setup

name = "qcopt"

description = ""

# README file as long_description.
long_description = io.open("README.md", encoding="utf-8").read()


# Read in requirements
requirements = open("requirements.txt").readlines()
requirements = [r.strip() for r in requirements]

qcopt_packages = ["qcopt"] + ["qcopt." + package for package in find_packages(where="qcopt")]

setup(
    name=name,
    url="",
    author="Teague Tomesh",
    author_email="teague@super.tech",
    python_requires=(">=3.8.0"),
    install_requires=requirements,
    extras_require={},
    license="N/A",
    description=description,
    long_description=long_description,
    packages=qcopt_packages,
    package_data={"qcopt": ["py.typed"]},
)

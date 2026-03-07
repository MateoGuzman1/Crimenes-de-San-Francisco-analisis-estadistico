#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from setuptools import find_packages, setup

NAME             = "sfcrime-model"
DESCRIPTION      = "Pipeline MLOps de clasificacion multiclase de crimenes en San Francisco."
URL              = "https://github.com/MateoGuzman1/Crimenes-de-San-Francisco-analisis-estadistico"
EMAIL            = "c.durangos@uniandes.edu.co"
AUTHOR           = "Equipo SF Crimes - MAIA"
REQUIRES_PYTHON  = ">=3.10.0"
long_description = DESCRIPTION

ROOT_DIR         = Path(__file__).resolve().parent
REQUIREMENTS_DIR = ROOT_DIR / "requirements"
PACKAGE_DIR      = ROOT_DIR / "package-src" / "sfcrime_model"

about = {}
with open(PACKAGE_DIR / "VERSION") as f:
    _version = f.read().strip()
    about["__version__"] = _version


def list_reqs(fname="requirements.txt"):
    with open(REQUIREMENTS_DIR / fname) as fd:
        return fd.read().splitlines()


setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    package_dir={"": "package-src"},
    packages=find_packages(where="package-src", exclude=("tests",)),
    package_data={
        "sfcrime_model": [
            "VERSION",
            "config.yml",
            "trained/*.pkl",
            "trained/label_classes.npy",
            "trained/__init__.py",
        ]
    },
    install_requires=list_reqs(),
    include_package_data=True,
    license="BSD-3",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
)

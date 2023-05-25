#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("CHANGELOG.rst") as changelog_file:
    changelog = changelog_file.read()

requirements = [
    "cls-python",
    "luigi"
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Jan Bessai",
    author_email="jan.bessai@tu-dortmund.de",
    python_requires=">=3.10",
    project_urls={
        "Bug Tracker": "https://github.com/cls-python/cls-luigi/issues",
        "Documentation": "TODO",
        "Source Code": "https://github.com/cls-python/cls-luigi",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Operating System :: MacOS",
    ],
    description="CLS-Luigi is an innovative pipeline tool designed to streamline the creation and execution of pipelines by harnessing the power of combinatory logic.",
    install_requires=requirements,
    license="Apache License (2.0)",
    long_description=readme + "\n\n" + changelog,
    include_package_data=True,
    keywords="cls-luigi",
    name="cls-luigi",
    packages=find_packages(include=["cls_luigi", "cls_luigi.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/cls-python/cls-luigi",
    version="0.1.0",
    zip_safe=False,
)

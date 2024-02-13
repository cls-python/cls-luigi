# -*- coding: utf-8 -*-
#
# Apache Software License 2.0
#
# Copyright (c) 2022-2023, Jan Bessai, Anne Meyer, Hadi Kutabi, Daniel Scholtyssek
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("CHANGELOG.rst") as changelog_file:
    changelog = changelog_file.read()

requirements = ["cls-python", "luigi", "requests"]

test_requirements = ["pytest>=3", "coverage"]

setup(
    author="Jan Bessai",
    author_email="jan.bessai@tu-dortmund.de",
    maintainer="Daniel Scholtyssek",
    maintainer_email="daniel.scholtyssek@tu-dortmund.de",
    python_requires=">=3.10",
    project_urls={
        "Bug Tracker": "https://github.com/cls-python/cls-luigi/issues",
        "Documentation": "https://cls-python.github.io/cls-luigi",
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
    description="CLS-Luigi is an innovative pipeline tool designed to streamline the creation and execution of algorithmic pipelines by harnessing the power of combinatory logic.",
    install_requires=requirements,
    license="Apache License (2.0)",
    long_description=readme + "\n\n" + changelog,
    include_package_data=True,
    keywords="cls-luigi",
    name="cls-luigi",
    packages=find_packages(
        include=["cls_luigi", "cls_luigi.*", "cls_luigi.visualizer.*"]
    ),
    test_suite="cls_luigi.tests",
    tests_require=test_requirements,
    url="https://github.com/cls-python/cls-luigi",
    version="0.1.0",
    zip_safe=False,
)

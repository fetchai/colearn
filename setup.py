# ------------------------------------------------------------------------------
#
#   Copyright 2021 Fetch.AI Limited
#
#   Licensed under the Creative Commons Attribution-NonCommercial International
#   License, Version 4.0 (the "License"); you may not use this file except in
#   compliance with the License. You may obtain a copy of the License at
#
#       http://creativecommons.org/licenses/by-nc/4.0/legalcode
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
import setuptools

keras_deps = [
    'tensorflow~=2.2.0',
    'tensorflow_datasets~=4.2.0',
    'tensorflow-privacy~=0.5.0',
]
other_deps = [
    'pandas~=1.1.0',
    'scikit-learn~=0.23.0',
]
pytorch_deps = [
    'opacus~=0.10.0',
    'Pillow~=8.0.1',
    'scikit-learn~=0.23.0',
    'scipy~=1.5.0',
    'torch~=1.7.0',
    'torchsummary~=1.5.0',
    'torchvision~=0.8.0',
]
docs_deps = [
    "mkdocs",
    "mkdocs-macros-plugin",
    "mkdocs-material",
    "mkdocs-material-extensions",
    "markdown-include",
]

grpc_deps = ['grpcio~=1.35.0',
             'grpcio-tools~=1.35.0',
             'prometheus_client==0.9.0',
             'click'
             ]
all_deps = list(set(keras_deps + other_deps + pytorch_deps + grpc_deps)) + ["xgboost"]

long_description = ""
try:
    with open("README.md", "r") as fh:
        long_description = fh.read()
except FileNotFoundError:
    print("README.md file not found, no long description available")

setuptools.setup(
    name="colearn",
    version="0.2.6",
    author="Fetch AI",
    author_email="developer@fetch.ai",
    description="The Standalone Fetch AI Collective Learning Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fetchai/colearn",
    packages=setuptools.find_namespace_packages(exclude=("tests", "tests.*", "site", "site.*",
                                                         "docs", "docs.*", "docker", "scripts", "build", "build.*")),
    classifiers=[
        # Need to fill in
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6, <3.9',
    install_requires=[
        'google-cloud-storage~=1.35.0',
        'matplotlib~=3.3.0',
        'numpy~=1.16.0',
        'pydantic~=1.7.0',
    ],
    tests_require=["tox~=3.20.0"],
    extras_require={
        'keras': keras_deps,
        'other': other_deps,
        'pytorch': pytorch_deps,
        'docs': docs_deps,
        'all': all_deps,
        'grpc': grpc_deps
    },
)

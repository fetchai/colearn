import setuptools

keras_deps = [
    'tensorflow~=2.3.0',
    'tensorflow_datasets~=4.2.0',
    'tensorflow-privacy~=0.5.0',
]
pytorch_deps = [
    'opacus~=0.10.0',
    'Pillow~=8.0.1',
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
             'prometheus_client==0.9.0',
             'click'
            ]
all_deps = keras_deps + pytorch_deps + grpc_deps

long_description = ""
try:
    with open("README.md", "r") as fh:
        long_description = fh.read()
except FileNotFoundError:
    print("README.md ile not found, no long description available")

setuptools.setup(
    name="colearn-interface-fetch-ai",
    version="0.0.1",
    author="Fetch AI",
    author_email="juan.besa@fetch.ai",
    description="The Standalone Fetch AI Collective Learning Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fetchai/colearn.git",
    packages=setuptools.find_packages(exclude=("tests", "examples", "docs")),
    classifiers=[
        # Need to fill in
        "Operating System :: OS Independent",
    ],
    python_requires='~=3.7',
    install_requires=[
        'google-cloud-storage~=1.35.0',
        'matplotlib~=3.3.0',
        'numpy~=1.16.0',
        'pandas~=1.1.0',
        'pydantic~=1.7.0',
        'scikit-learn~=0.23.0',
    ],
    tests_require=["tox~=3.20.0"],
    extras_require={
        'keras': keras_deps,
        'pytorch': pytorch_deps,
        'docs': docs_deps,
        'all': all_deps,
        'grpc': grpc_deps
    },
)

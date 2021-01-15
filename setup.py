import setuptools

keras_deps = ['tensorflow']
pytorch_deps = ['torch']
docs_deps = ["mkdocs", "mkdocs-macros-plugin", "mkdocs-material",
             "mkdocs-material-extensions", "markdown-include"]
examples_deps = ['pandas',
                 'sklearn',
                 'scikit-learn',
                 'torchsummary',
                 'scipy',
                 'opencv-python']

all_deps = keras_deps + pytorch_deps + examples_deps

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="colearn-interface-fetch-ai",
    version="0.0.1",
    author="Fetch AI",
    author_email="juan.besa@fetch.ai",
    description="The Standalone Fetch AI Collective Learning Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fetchai/colearn.git",
    packages=setuptools.find_packages(),
    classifiers=[
        # Need to fill in
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['matplotlib',
                      'pydantic',
                      'numpy'
                      ],
    tests_require=["tox"],
    extras_require={
        'keras': keras_deps,
        'pytorch': pytorch_deps,
        'docs': docs_deps,
        'examples': examples_deps,
        'all': all_deps
    },
)

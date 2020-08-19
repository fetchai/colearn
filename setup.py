import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="colearn-interface-fetch-ai",
    version="0.0.1",
    author="Fetch AI",
    author_email="juan.besa@fetch.ai",
    description="The Collective Learning Interface to access the Fetch AI Collective Learning Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fetchai/colearn-interface.git",
    packages=setuptools.find_packages(),
    classifiers=[
        # Need to fill in
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

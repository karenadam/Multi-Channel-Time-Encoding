import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="timeEncoding",
    version="0.0.1",
    author="Karen Adam",
    author_email="karen.adam@epfl.ch",
    description="A package that performs time encoding and reconstruction of multiple signals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/karenadam/Multi-Channel-Time-Encoding",
    project_urls={"Bug Tracker": "https://github.com/pypa/sampleproject/issues"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)

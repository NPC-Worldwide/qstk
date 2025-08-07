from setuptools import setup, find_packages

setup(
    name="qstk",
    version="0.1.0",
    description="Quantum Semantic Toolkit: Quantum semantic methods powered by npcpy.",
    author="Christopher Agostino",
    author_email="info@npcworldwi.de",
    url="https://npcworldwi.de",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "npcpy>=1.1.7"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="quantum semantics npcpy toolkit",
    license="MIT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    project_urls={
        "Company": "https://enpisi.com",
    },
)

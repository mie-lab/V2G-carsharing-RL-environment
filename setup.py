"""Package installer."""
import os
from setuptools import setup
from setuptools import find_packages

LONG_DESCRIPTION = ""
if os.path.exists("README.md"):
    with open("README.md") as fp:
        LONG_DESCRIPTION = fp.read()

scripts = []

setup(
    name="rl_v2g",
    version="0.0.1",
    description="Reinforcement learning environment for V2G for carsharing",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="MIE Lab",
    author_email=("nwiedemann@ethz.ch"),
    license="GPLv3",
    url="https://github.com/mie-lab/v2g_thesis",
    install_requires=["numpy", "scipy", "pandas", "gymnasium"],
    classifiers=[
        "License :: OSI Approved :: MIT",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages("."),
    python_requires=">=3.8",
)
from setuptools import setup, find_packages

setup(
    name="covmadt",
    version="0.1.0",
    author="Research Team",
    author_email="research@example.com",
    description="Implementation of CovMADT: Efficient Offline Multi-Agent Reinforcement Learning via Convex Markov Games",
    long_description=open("README.md").read() if __import__('os').path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/covmadt",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines() if line.strip() and not line.startswith("#")
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pylint>=2.17.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
)



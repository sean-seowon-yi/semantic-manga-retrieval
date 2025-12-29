#!/usr/bin/env python3
"""
Setup script for manga-vectorizer package.

Install in development mode:
    pip install -e .

Install normally:
    pip install .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().split("\n")
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="manga-vectorizer",
    version="1.0.0",
    description="Manga character retrieval using CLIP embeddings and LLM descriptions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sean Yi",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "manga-eval-qcfr=manga_vectorizer.evaluation.recall_qcfr:main",
            "manga-eval-fusion=manga_vectorizer.evaluation.recall_fusion:main",
            "manga-eval-image=manga_vectorizer.evaluation.recall_image:main",
            "manga-eval-text=manga_vectorizer.evaluation.recall_text:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords="manga, anime, character retrieval, CLIP, embeddings, FAISS",
)

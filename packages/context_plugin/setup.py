"""
Setup script for Claude Code Context Plugin
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="context_plugin",
    version="0.1.0",
    description="Claude Context Plugin Monorepo Package",
    author="Your Name",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[],  # Add dependencies here
    include_package_data=True,
    python_requires=">=3.8",
)

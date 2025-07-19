from setuptools import setup, find_packages

setup(
    name="groq_cursor",
    version="0.1.0",
    description="Groq Cursor Integration Package",
    author="Your Name",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[],  # Add dependencies here
    include_package_data=True,
    python_requires=">=3.8",
)

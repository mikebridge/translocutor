# setup.py

from setuptools import setup, find_packages

setup(
    name="translocutor",
    version="0.0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'translocutor=translocutor.cli:main',
        ],
    },
    install_requires=[
        # List any dependencies here, e.g., 'click', 'typer', etc.
        'openai',
        'orjson',
        'pydantic',
        'python-dotenv',
        'tiktoken',
        'webvtt-py'
    ],
    description="A simple command-line tool.",
    author="Mike Bridge",
    author_email="mike@bridgecanada.com",
    url="https://github.com/mikebridge/translocutor",  # Update with your repo URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)

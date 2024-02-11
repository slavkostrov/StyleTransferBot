#!/usr/bin/python3.8

import setuptools

setuptools.setup(
    name="TransferBot",
    version="1.0.0",
    author="Slava Kostrov",
    author_email="slavkotrov@gmail.com",
    description="Telegram bot for style transfer",
    # long_description=,
    long_description_content_type="text/markdown",
    # url=,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # install_requires=[],
    python_requires=">=3.8",
)

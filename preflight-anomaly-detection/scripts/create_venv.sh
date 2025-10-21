#!/bin/bash

# Create a virtual environment for the project
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the required packages
pip install -r ../requirements.txt

echo "Virtual environment created and packages installed."
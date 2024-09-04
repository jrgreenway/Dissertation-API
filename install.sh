#!/bin/bash

if ! command -v python &> /dev/null
then
    echo "Python is not installed. Please install Python and try again."
    exit 1
fi

python -m venv venv
if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment."
    exit 1
fi

source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment."
    exit 1
fi

pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies."
    exit 1
fi

echo "Virtual environment setup complete."
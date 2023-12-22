#!/bin/bash
PIP_PATH="/app/venv/bin/pip"
REQUIREMENTS_FILE="requirements.txt"
$PIP_PATH install --upgrade pip
$PIP_PATH install --upgrade -r $REQUIREMENTS_FILE
echo "Sucessfully install all dependencies!"

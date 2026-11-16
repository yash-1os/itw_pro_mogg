#!/bin/bash

# Check if a path was provided as an argument
if [ -z "$1" ]; then
  echo "Error: Please provide the path to your project folder."
  echo "Usage: ./setup-git.sh /path/to/your/project"
  exit 1
fi

PROJECT_PATH="$1"

# Verify the directory exists
if [ ! -d "$PROJECT_PATH" ]; then
  echo "Error: The directory '$PROJECT_PATH' does not exist."
  exit 1
fi

# Navigate to the folder and run Git commands
cd "$PROJECT_PATH" || exit

echo "Initializing Git repository in $PROJECT_PATH..."
git init
git add .
git commit -m "Initial commit: Behavior analysis project"

echo "Success: Local Git repository initialized."

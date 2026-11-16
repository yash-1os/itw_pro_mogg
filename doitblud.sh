#!/bin/bash

# Check if a path was provided as an argument
if [ -z "$1" ]; then
  echo "Error: Please provide the path to your project folder."
  echo "Usage: ./doitblud.sh /path/to/your/project"
  exit 1
fi

PROJECT_PATH="$1"
# INSERT YOUR COPIED GITHUB URL HERE (Keep the quotes):
GITHUB_URL="https://github.com/yash-1os/itw_pro_mogg.git"

# Verify the directory exists
if [ ! -d "$PROJECT_PATH" ]; then
  echo "Error: The directory '$PROJECT_PATH' does not exist."
  exit 1
fi

cd "$PROJECT_PATH" || exit

echo "Preparing files..."
git init
git add .

# Commit changes (ignores the error if there's nothing new to commit)
git commit -m "Auto-update: Syncing project files to GitHub" || echo "No new changes to commit."

# Link to GitHub and force the master branch
git remote remove origin 2>/dev/null
git remote add origin "$GITHUB_URL"
git branch -M master

echo "Uploading to GitHub..."
git push -u origin master

echo "Success: Process complete!"
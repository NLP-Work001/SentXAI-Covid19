#!/usr/bin/env bash

# Create processed folder only if it does not exist in the folder tree.
FILE_DIR="data/processed"

if [[ ! -d "$FILE_DIR" ]]; then
    echo "Create '$FILE_DIR' folder ..."
    mkdir -p "${FILE_DIR}"
else
    echo "Folder '$FILE_DIR' already exists."
fi
#!/usr/bin/env bash

declare -r processed_dir="data/processed"
declare -r split_dir="data/split"
declare -r run_script="src/prepare.py"

if [[ ! -d "${processed_dir}" ]]; then
        echo "Make sure the file folder '${processed_dir}' exists!"
        exit
fi

if [[ ! -d "${split_dir}" ]]; then
        echo "Creating file folder '${split_dir}'."
        mkdir -p "${split_dir}"
fi

echo "Running ${run_script} script ..."
python "${run_script}" "${processed_dir}" "${split_dir}"
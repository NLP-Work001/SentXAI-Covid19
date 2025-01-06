#!/usr/bin/env bash

declare -r split_dir="data/split"
declare -r run_script="src/cross_valid.py"
declare -r model_out="models/"

date_time=$(date +"%Y-%m-%d %H:%M:%S")

if [[ ! -d "${split_dir}" ]]; then
	echo "Make sure the file folder '${split_dir}' exists!"
	exit
fi

if [[ ! -f "${run_script}" ]]; then
	ehco "File name '$run_script' does not exist!"
	exit
fi

if [[ ! -d "$model_out" ]]; then
	echo "Creating model outputs folder '$model_out'."
	mkdir -p "$model_out"
fi 

echo "Running ${run_script} script ..."
python "${run_script}" -o "${model_out}" -s "${split_dir}" -d "${date_time}"

#!/usr/bin/env bash

usage() {
	echo "USAGE: exactly two args are required .i.e <script>.py and directory."
	exit
}

[[ "$#" -ne 2 ]] && usage

declare -r run_script="src/$1"
declare -r dir_out_="$2"
date_time=$(date +"%Y-%m-%d %H:%M:%S")

if [[ ! -f "${run_script}" ]]; then
	echo "File name '$run_script' does not exist!"
	exit
fi

# if [[ ! -d "$dir_out_" ]]; then
#     echo "Creating model output folder..."
#     mkdir -p "$dir_out_"
# fi

echo "Executing ${run_script} script ..."
python "$run_script" -d "$date_time" -o "$dir_out_"

# Execute script to filter best parameters
# model_folder=$(echo "$run_python" | grep -E -o "[^folder:].*")
# model_out=$(echo "$run_python" | grep -E -o "[^moodel:].*")
# echo "Running filtering params script ..."
# ./scripts/filter_params.sh "$model_folder" "$model_out"

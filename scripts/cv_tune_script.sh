#!/usr/bin/env bash

# Script usage: ./scripts/cv_tune_script.sh <python-script-name>

usage() {
	echo "USAGE: exactly one argument is needed i.e. method <arg>.py"
	exit
}

[[ "$#" -ne 1 ]] && usage

declare -r file_dir_="data/split"
declare -r model_out_="models/"
date_time=$(date +"%Y-%m-%d %H:%M:%S")
script="src/$1"

is_valid() {
	echo "USAGE: exactly two arguments are needed .i.e method <arg1> <arg2>"
	exit
}

validation_func() {

	echo "Number of args: $#"

	[[ "$#" -ne 2 ]] && is_valid

	local first_="$1"
	local second_="$2"

	if [[ ! -d "${first_}" ]]; then
		echo "Make sure the file folder '${first_}' exists!"
		exit
	fi

	if [[ ! -f "${second_}" ]]; then
		echo "File name '$second_' is not a file!"
		exit
	fi
}

validation_func "$file_dir_" "$script" && echo "Good to go!"

if [[ ! -d "$model_out_" ]]; then
	echo "Creating model outputs folder '$model_out_'."
	mkdir -p "$model_out_"
fi

echo "Running ${script} script ..."
python "${script}" -o "${model_out_}" -s "${file_dir_}" -d "${date_time}"

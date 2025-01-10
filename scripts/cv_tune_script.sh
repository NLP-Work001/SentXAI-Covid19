#!/usr/bin/env bash

usage() {
	echo "USAGE: single python script as argument is required."
	exit
}

[[ "$#" -ne 1 ]] && usage

declare -r run_script="src/$1"
date_time=$(date +"%Y-%m-%d %H:%M:%S")

if [[ ! -f "${run_script}" ]]; then
	ehco "File name '$run_script' does not exist!"
	exit
fi

echo "Executing ${run_script} script ..."
python "$run_script" -d "$date_time"
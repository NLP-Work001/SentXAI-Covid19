#!/usr/bin/env bash

declare -r run_script="src/cross_valid.py"
date_time=$(date +"%Y-%m-%d %H:%M:%S")

if [[ ! -f "${run_script}" ]]; then
	ehco "File name '$run_script' does not exist!"
	exit
fi

echo "Executing ${run_script} script ..."
python "$run_script" -d "$date_time"

#!/usr/bin/env bash

declare -r file_name="src/$1"

usage() {
	echo "USAGE: $0 [Only single argument is required.]"
	exit
}

[[ $# -eq 0 || $# -gt 2 ]] && usage

if [[ ! -e "$file_name" ]]; then
	echo "File name '$file_name' does not exist!"
	exit
fi

black "$file_name"
isort format "$file_name"
ruff format "$file_name"

pylint "$file_name"
flake8 "$file_name"

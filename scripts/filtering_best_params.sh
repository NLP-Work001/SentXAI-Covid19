#!/usr/bin/env bash

# Constant variables
# folder="logistic"

if [[ ! $# -eq 2 ]]; then
    echo "USAGE: exactly two args are required!"
    exit 1
fi

declare -r path="$1"
declare -r  model_name="$2"
declare -r json_output="${path}/best_params.json"

find_params_score() {
	local filter_values=("columntransformer__tfidfvectorizer__" "${model_name}__")
	local best_params_=()

	# Extract relevant parameters using grep and sed
	for name in "${filter_values[@]}"; do
		best_params_+=("$(grep "$name" "$1" | sed "s/$name//g" | sed "s/\"//g" | head -1)")
	done
	# Clean up ngram_range parameter
	pattern="np\.*[a-z]*[0-9].?\(|\)"
	best_params_[0]=$(echo "${best_params_[0]}" | grep "ngram_range" | sed -E "s/\(/\[/g" | sed -E "s/\)/\]/g")

	# Clean up other parameters
	best_params_[1]=$(echo "${best_params_[1]}" | grep -v "ngram_range" | sed -E "s/$pattern//g")

	# Extract scores and format parameters
	new_arr=()
	scores=()
	for c in $(seq 0 1); do
		value1=$(echo "${best_params_[$c]}" | grep -o "},.*" | cut -d, -f2)
		value2=$(echo "${best_params_[$c]}" | grep -o "},.*" | cut -d, -f3)
		params=$(echo "${best_params_[$c]}" | grep -o ".*}")
		scores+=($value1 $value2)
		new_arr[$c]=$(echo "{\"params\": $params, \"f1_weighted\": $value1, \"roc_auc_ovo\": $value2}" | sed "s/'/\"/g")
	done

	# Combine parameters for vectorizer and model
	new_arr[0]="{\"vectorizer\": ${new_arr[0]}, "
	new_arr[1]="\"model\": ${new_arr[1]}}"

	# Calculate average score
	sum=0
	count=${#scores[@]}
	for c in ${scores[@]}; do
		sum=$(echo "$c + $sum" | bc)
	done
	avg_score=$(echo "scale=2; $sum / $count " | bc)

	echo "score: ${avg_score}"
	echo "${new_arr[@]}"
}

# Find the latest optimized parameters file
readarray -t files < <(find "$path" -maxdepth 1 -name "*.csv")
echo "${files[0]}" | cut -d/ -f3
if [[ ! ${#files[@]} -eq 0 ]]; then

	echo "Folder is not empty: ${#files[@]}."
	echo "$params_"
	params_=$(find_params_score ${files[0]} | grep -v "score:")
	# new_file=$(echo "${files[0]}" | cut -d/ -f3)
	echo ${params_[@]} | python -m json.tool >$json_output
	# echo $new_file >"${path}/log.txt"
else
	echo "There are not existing *.csv files."
	exit
fi

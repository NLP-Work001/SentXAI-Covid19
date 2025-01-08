#!/usr/bin/env bash

# Constant variables
declare -r path="models/logisticregression"
declare -r json_output="${path}/best_params.json"
declare -r log_file="${path}/best_param_scores_.txt"

# Find the latest optimized parameters file
file_name=$(find "$path" -name "*.csv" | sort -t_ -k3,3 -k4,4 -r | head -1)

# Filter and process parameter values
filter_values=('columntransformer__tfidfvectorizer__' 'logisticregression__')

> "$log_file"
for name in "${filter_values[@]}"; do
  grep "$name" "$file_name" | head -1 | sed "s/$name//g" | sed "s/\"//g" >> "$log_file"
done

# Extract formatted output form python script and overrride `log_file`
py_out=$(python src/re_filter.py "$path")
echo "$py_out" > $log_file
# echo $py_out
# Extract values for JSON output
readarray -t params < <(grep -o ".*}" "$log_file" | sed "s/'/\"/g" | sed "s/(/[/g" | sed "s/)/]/g")
readarray -t metrics < <(grep -o "},.*" "$log_file" | cut -d, -f2,3)
# echo "${params[@]}"

vect_params="${params[0]}"
model_params="${params[1]}"
vectorizer_f1=$(echo "${metrics[0]}" | cut -d, -f1)
vectorizer_roc_auc=$(echo "${metrics[0]}" | cut -d, -f2)
model_f1=$(echo "${metrics[1]}" | cut -d, -f1)
model_roc_auc=$(echo "${metrics[1]}" | cut -d, -f2)

# JSON File
cat << EOF > "$json_output"
[{
  "vectorizer": {
    "params": $vect_params,
    "roc_auc_ovo": $vectorizer_roc_auc,
    "f1_weighted": $vectorizer_f1
  },
  "model": {
    "params": $model_params,
    "roc_auc_ovo": $model_roc_auc,
    "f1_weighted": $model_f1
  }
}]
EOF

# cat $json_output

# Folder clean-up
rm -rf "$log_file"

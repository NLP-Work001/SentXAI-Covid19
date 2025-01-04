#!/usr/bin/env bash

FILE_TO_LOG="covid19_tweet.csv"
FILE_FOLDER="data/processed"

if [[ -f "${FILE_FOLDER}/${FILE_TO_LOG}" ]]; then
  echo "$FILE_TO_LOG already loaded to $FILE_FOLDER"
else
  echo "File name: ${FILE_TO_LOG} does not exist in ${FILE_FOLDER}"
fi
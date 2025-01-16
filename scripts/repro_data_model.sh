#!/usr/bin/env bash
model_out="$1"
# dvc stage add --force -n process \
#     -p config.yml:data.processed.data_size \
#     -d src/process.py -d data/raw/covid-19-tweets \
#     -o data/processed \
#     python src/process.py data/raw/covid-19-tweets data/processed/processed_tweets.csv


# dvc stage add --force -n split \
#     -p config.yml:data.split.ratio,data.split.seed \
#     -d src/split.py -d data/processed \
#     -o data/split \
#     python src/split.py data/processed data/split


# dvc stage add --force -n eda \
#     -p config.yml:data.processed.data_size \
#     -d src/ed_analytics.py -d data/processed \
#     -o ed-analytics/ \
#     python src/ed_analytics.py data/processed ed-analytics/


# dvc stage add --force -n dev \
#     -p config.yml:models.seed \
#     -d src/model_dev.py \
#     -o models/central/ \
#     python src/model_dev.py models/central/

# Dynamically set a model output
dvc stage add --force -n train \
    -p config.yml:models.train.model \
    -d scripts/run_model_script.sh \
    -o models/trained/"$model_out" \
    ./scripts/run_model_script.sh model_train.py models/trained/

dvc stage add --force -n cross_valid \
        -p config.yml:models.cross_validation.model,models.cross_validation.n_splits \
        -d scripts/run_model_script.sh \
        -o models/cross_valid/"$model_out"\
        ./scripts/run_model_script.sh model_cross_valid.py


dvc stage add --force -n hyper_tuner \
    -p config.yml:models.fine_tune.model,models.fine_tune.n_splits \
    -d scripts/run_model_script.sh -d data/split \
    -o models/tuned/"$model_out"\
    ./scripts/run_model_script.sh hyper_tuner.py

# Data model repro
dvc repro



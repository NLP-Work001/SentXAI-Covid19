#!/usr/bin/env bash
# echo  "Creating data pipeline"
# dvc stage  add -f -n process \
#                 -p configs.yml:data.processed.data_size \
#                 -d src/data_processing/data_preprocessing.py -d data/raw/covid-19-tweets \
#                 -o data/processed \
#                 python src/data_processing/data_preprocessing.py data/raw data/processed


echo "Creating EDA stage ..."

dvc stage add -f -n data_analysis \
        -d src/data_processing/exploratory_analysis.py -d data/processed \
        -o data/analytics \
        python src/data_processing/exploratory_analysis.py data/processed data/analytics

# echo "Creating data split stage ..."

# dvc stage add --f -n data_split \
#     -p configs.yml:data.split.ratio,data.split.seed \
#     -d src/data_processing/data_splitting.py -d data/processed \
#     -o data/split \
#     python src/data_processing/data_splitting.py data/processed data/split

# echo "Creating baseline models stage ..."

# dvc stage add -f -n baseline_model \
#     -p configs.yml:models.seed \
#     -d src/model_development/baseline_model.py \
#     -o src/model_checkpoints/baselines \
#     python src/model_development/baseline_model.py src/model_checkpoints/baselines


# echo "Creating training model stage ..."

# dvc stage add -f -n training \
#     -p configs.yml:models.train.model \
#     -d src/model_development/model_training.py -d data/split \
#     -o src/model_checkpoints/trained \
#     python src/model_development/model_training.py data/split src/model_checkpoints/trained

# echo "Creating cross validation stage ..."

# dvc stage add -f -n cross_validation \
#         -p configs.yml:models.cross_validation.model,models.cross_validation.n_split \
#         -d src/model_optimization/cross_validation.py -d data/split \
#         -o src/model_checkpoints/cross_valid \
#         python src/model_optimization/cross_validation.py data/split src/model_checkpoints/cross_valid

# echo "Creating hyperparameter tuning stage ..."

# dvc stage add -f -n hyper_optimizer \
#     -p configs.yml:models.train.optimize \
#     -d src/model_optimization/hyper_opt.py -d data/split \
#     -o src/model_checkpoints/tuned \
#     python src/model_optimization/hyper_opt.py data/split

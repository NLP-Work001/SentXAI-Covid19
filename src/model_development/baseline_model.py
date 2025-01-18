import os
import joblib
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
sys.path.append(str("src/helpers"))
from utils import load_parameters
from helpers import config_loader

def main() -> None:

    # File paths
    parent_ = config_loader("configs.yml")
    file_name_ = parent_["models"]["baseline"]["file"]
    seed_ = parent_["models"]["seed"]
    os.makedirs(sys.argv[1], exist_ok=True)
    
    # Collection of all models
    models = {
        "bayes": MultinomialNB(),
        "svc": SVC(probability=True),
        "logistic": LogisticRegression(random_state=seed_, max_iter=1000),
        "forest": RandomForestClassifier(random_state=seed_),
        "tree": DecisionTreeClassifier(random_state=seed_),
    }

    # Save models into pickle files
    for name, model in models.items():
        out_ = Path(sys.argv[1]) / f"{name}_{file_name_}"
        with open(out_, "wb") as f:
            joblib.dump(model, f)
            print(f"{name} model has been saved into {file_name_}.")

if __name__ == "__main__":
    main()
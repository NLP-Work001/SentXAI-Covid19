import os
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from utils import load_parameters

def main() -> None:

    # File paths
    parent_ = load_parameters("config.yml")
    dev_ = parent_["models"]["dev"]
    seed_ = parent_["models"]["seed"]
    path_out_ = dev_["path"]
    file_out_ = dev_["file"]
    os.makedirs(path_out_, exist_ok=True)
    
    
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
        out_ = Path(path_out_) / f"{name}_{file_out_}"
        with open(out_, "wb") as f:
            joblib.dump(model, f)
            print(f"{name} model has been saved into {path_out_}.")

if __name__ == "__main__":
    main()
import joblib

if __name__ == "__main__":
    model = joblib.load("models/logisticregression/model.pkl")
    
    print(model.__class__.__name__)
    print(model.get_params())
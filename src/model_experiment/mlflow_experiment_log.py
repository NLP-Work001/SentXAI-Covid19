import mlflow
from mlflow.models import infer_signature


# ToDO: Complete the class method ...

class MLFlowExperiment:

    def __init__(self, *args, **kwargs) -> None:
        # Create Experiment
        self.experiment_name = f"SentXAI NLP Experiment"
        self.experiment_details = mlflow.get_experiment_by_name(self.experiment_name)

        if not experiment_details:
            # Experiment Description
            description = (
                f"Sentiment Analysis for covid19 tweets dataset."
                "Various scikit-learn machine learning classifers are experimented."
            )

            experiment_tags = {
                "project-name": "SentXAI project",
                "mlflow.note.content": description,
            }

            mlflow.create_experiment(name=self.experiment_name, tags=experiment_tags)
            print("Experiment Created!")

        # Get the experiment ID
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        self.experiment_id = experiment.experiment_id


# Log Experiment Runs
def track_model_experiment(self, experiment_id, model, X_train, params, metrics) -> None:

    name =  model.__class__.__name__.lower()

    with mlflow.start_run(experiment_id=experiment_id, run_name=name) as run:

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(
            f"reports/{model.__class__.__name__}_cls_report.csv",
            artifact_path=name,
        )
        mlflow.log_artifact(
            f"reports/{name}_cls_report.csv",
            artifact_path=name,
        )

        mlflow.log_artifact("reports/vectorizer.pkl", artifact_path=name)
        # mlflow.log_figure(figure, f"{name}/{model.__class__.__name__}_cm.png")

        model_signature = infer_signature(X_train[:10], model.predict(X_train[:10]))
        mlflow.sklearn.log_model(sk_model=model, infer_signature=model_signature, artifact_path=name, registered_model_name=name)

        # Log model run_id history
        model_run_id = f"Model: {name}, RUN_ID: {run.info.run_id}"



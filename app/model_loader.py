import mlflow.pyfunc

def load_model():
    model_uri = "models:/model/Production"
    return mlflow.pyfunc.load_model(model_uri)
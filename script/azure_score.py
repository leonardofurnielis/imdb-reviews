import os
import json
import joblib

import pandas as pd

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    model_path = os.path.join(str(os.getenv("AZUREML_MODEL_DIR")), "azure_credit_risk_model.pickle")
    model = joblib.load(model_path)


def run(input_payload):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    try:
        if type(input_payload) is str:
            dict_data = json.loads(input_payload)
        else:
            dict_data = input_payload

        data = pd.DataFrame.from_dict(dict_data['input'])
        predictions = model.predict(data)
        scores = model.predict_proba(data)
        records = []

        for pred, proba in zip(predictions, scores):
            records.append({ "Scored Labels": pred.tolist(), "Scored Probabilities": proba.tolist() })

        return { "output": records }
        
    except Exception as e:
        result = str(e)
        return { "error": result }

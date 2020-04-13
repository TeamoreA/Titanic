
from flask import Flask, request
import pandas as pd
import numpy as np
import json
import pickle
import os

app = Flask(__name__)

model_path = os.path.join(os.path.pardir, os.path.pardir, 'models')
model_file_path = os.path.join(model_path, 'lf_model.pkl')
scaler_file_path = os.path.join(model_path, 'lf_scaler.pkl')
with open(model_file_path, 'rb') as model_file:
    model = pickle.load(model_file)
with open(scaler_file_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
    
columns = [
    u'Age', u'Fare', u'FamilySize', u'IsMother', u'IsMale', u'Deck_A', u'Deck_B'\
    u'Deck_C', u'Deck_D', u'Deck_E', u'Deck_F', u'Deck_G', u'Deck_Z', u'Pclass_1',\
    u'Pclass_2', u'Pclass_3', u'Title_Lady', u'Title_Master', u'Title_Miss', u'Title_Mr',\
    u'Title_Mrs', u'Title_Officer', u'Title_Sir', u'Fare_Bin_very_low', u'Fare_Bin_low',\
    u'Fare_Bin_very_high', u'Fare_Bin_high', u'Embarked_C', u'Embarked_Q', u'Embarked_S',\
    u'AgeState_Adult', u'Agestate_Child'
]

@app.route('/api', methods=['POST'])
def make_prediction():
    data = json.dumps(request.get_json(force=True))
    df = pd.read_json(data)
    passenger_ids = df['PassengerId'].ravel()
    actuals = df['Survived'].ravel()
    X = df.columns.to_numpy(dtype='float')
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    res = pd.DataFrame({'PassengerId': passenger_ids, 'Predicted': predictions, 'Actual': actuals})
    return res.to_json()

if __name__ == "__main__":
    app.run(port=10001, debug=True)
    

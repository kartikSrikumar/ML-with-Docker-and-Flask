  
import pandas as pd
from flask import Flask, jsonify, request
import joblib

app = Flask(__name__)

#request.post('/predict', json={'data'})
@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json(force=True)
    input_data = req['data']
    input_data_df = pd.DataFrame.from_dict(input_data)

    model = joblib.load('model.pkl')
    # scale_obj = joblib.load('scale.pkl')

    # input_data_scaled = scale_obj.transform(input_data_df)

    # print(input_data_scaled)

    prediction = model.predict(input_data_df)

    if prediction[0] == 1:
        Fraud = 'Fraudulant Transaction'
    else:
        Fraud = 'Safe Transaction'

    return jsonify({'output':{'Fraud':Fraud}})
        

@app.route('/')
def home():
    return "Welcome to Block Watch!"


if __name__=='__main__':
    app.run(host='0.0.0.0', port='3000')


 
# {
#     "data": [
#         {
#             "Avg min between sent tnx": 106,
#             "Avg min between received tnx": 199,
#             "avg val received": 6,
#             "avg val sent": 3
#         }
#     ]
# }
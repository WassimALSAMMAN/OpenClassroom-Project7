#import joblib
import xgboost as xgb 
from flask import Flask, jsonify, request
import pandas as pd 

app = Flask(__name__)


X_test_dashboard = pd.read_csv('./X_test_dashboard.csv')
# Remove joblib because it doesn't work with streamlit cloud
#filename = "model_xgb_dashb.joblib"
#loaded_model = joblib.load(filename)
model_xgb = xgb.XGBClassifier(
    learning_rate = 0.13340120276728681,
    max_depth = 43,
    min_child_weight = 1.5589983674569985,
    reg_alpha = 0.08048825852598339,
    reg_lambda = 0.006187471945504702,
    seed = 42
)
X_train_dashboard = pd.read_csv('./X_train_dashboard.csv').drop(columns=['SK_ID_CURR'])
y_train_dashboard = pd.read_csv('./y_train_dashboard.csv').drop(columns=['SK_ID_CURR'])
loaded_model = model_xgb.fit(X_train_dashboard, y_train_dashboard)
limit_proba = 0.8


# To return the client score
@app.route('/client_score', methods=['POST'])
def client_score():
    '''_Use the function client_score() to get the probability of a client by his number_.
    Args:
        none.  
                    
    Returns:
        proba: _Client score by his number_.
    '''
    number_client = request.get_json()
    number_client = number_client['number_client']
    X_test = X_test_dashboard.loc[X_test_dashboard['SK_ID_CURR']==number_client].drop(columns=['SK_ID_CURR'])
    proba = loaded_model.predict_proba(X_test)[0][0]
    return jsonify(float(proba))


# To return the client status
@app.route('/client_status', methods=['POST'])
def client_status():
    '''_Use the function client_status() to get the validation status(Credit is accepted or not)_.
    Args:
        none.  
                    
    Returns:
        client_status_result: _Client status by his score. If the score < 0.8, credit is not accepted. 
                                    If the score > 0.8, credit is accepted.
    '''
    client_score = request.get_json()
    client_score = client_score['client_score']
    if client_score > limit_proba:
        client_status_result = 'Good client, credit is accepted'    
    else:
        client_status_result = 'Bad client, credit is not accepted'    
    return jsonify(client_status_result)


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=6000,debug=True)
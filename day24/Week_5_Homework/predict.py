# Load the Model

#importing pickle library
import pickle

#importing flask libriary and modules from flask
from flask import Flask
from flask import request
from flask import jsonify



model_file = 'model1.bin'


with open(model_file,'rb') as f_in:#'rb' - Read binary
    model = pickle.load(f_in)

dv_file = 'dv.bin'

with open(dv_file,'rb') as f_in:#'rb' - Read binary
    dv = pickle.load(f_in)
    
app = Flask('Churn')#Churn service


@app.route('/Predict',methods=['POST'])#maps the specific URL with the associated function that is intended to perform some task. 


#POST - Used to create or add a resource on the server.POST is a request method supported by HTTP.
#The HTTP POST method requests the web server accept the data enclosed in the body of the POST message. 
#HTTP POST method is often used when submitting login or contact forms or uploading files and images to the server.


def predict():

  customer = request.get_json()

  X = dv.transform([customer])
  y_pred = model.predict_proba(X)[:,1][0]
  churn = y_pred>=0.5

  result = {
    'Churn_probability': float(y_pred),
    'Churn': bool(churn)
  }

  return jsonify(result) #jsonify serialized data to JSON format,wrapts it up in a Response boject with the json mimetype.It properly returns JSON data.
                        #It returns a Response object   



if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=9696)
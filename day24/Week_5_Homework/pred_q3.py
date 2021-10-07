# Load the Model

#importing pickle library
import pickle



model_file = 'model1.bin'


with open(model_file,'rb') as f_in:#'rb' - Read binary
    model = pickle.load(f_in)

dv_file = 'dv.bin'

with open(dv_file,'rb') as f_in:#'rb' - Read binary
    dv = pickle.load(f_in)
    
customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 19.7}


X = dv.transform([customer])

y_pred = model.predict_proba(X)[:,1][0]

print('Input : ',customer)
print("Churn probability : ",y_pred)


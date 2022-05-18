from flask import Flask,request,jsonify
import pickle
import numpy as np

model = pickle.load(open('model.pk1','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict',methods=['POST'])

def predict():
    Gender = request.form.get('Gender')
    Married = request.form.get('Married')
    Dependents = request.form.get('Dependents')
    Education = request.form.get('Education')
    Self_Employed = request.form.get('Self_Employed')
    ApplicantIncome = request.form.get('ApplicantIncome')
    CoapplicantIncome = request.form.get('CoapplicantIncome')
    LoanAmount = request.form.get('LoanAmount')
    Loan_Amount_Term = request.form.get('Loan_Amount_Term')
    Credit_History = request.form.get('Credit_History')
    Property_Area = request.form.get('Property_Area')

    # result = {'Gender': Gender, 'Married': Married, 'Dependents': Dependents, 'Education': Education,
    #           'Self_Employed': Self_Employed, 'ApplicantIncome': ApplicantIncome,'CoapplicantIncome':CoapplicantIncome,'LoanAmount':LoanAmount,'Loan_Amount_Term':Loan_Amount_Term,'Credit_History':Credit_History,'Property_Area':Property_Area}

    input_query = np.array([[Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area]])

    result = model.predict(input_query)[0]

    return jsonify({'Loan_Status':str(result)})


if __name__ == '__main__':
    app.run(debug=True)
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
#import sklearn

#Create The App
app = Flask(__name__)
model = pickle.load(open("dt_with_hyp.pkl", "rb"))


#Home Path
@app.route("/")
def Home():
    return render_template("home.html")

#Prediction
@app.route("/predict", methods = ["POST"])
def predict():
    ip_features = [float(x) for x in request.form.values()]
    features = [np.array(ip_features)]
    #features = np.append(features,[0,0,0])
    prediction = model.predict(features)
    #prediction.reshape(-1,1)
    return render_template("home.html", prediction_text = "The insurance is {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
    
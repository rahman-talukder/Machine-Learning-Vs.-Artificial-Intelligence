from flask import Flask, render_template, request
import numpy as np
import pickle
from sklearn.svm import SVC

app = Flask(__name__)

load_model = pickle.load(open("svm_model_for_something.sav", "rb"))


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        Pclass = int(request.form["Pclass"])
        Age = float(request.form["Age"])
        SibSp = int(request.form["SibSp"])
        Parch = int(request.form["Parch"])
        Fare = float(request.form["Fare"])
        encoded_Embarked = int(request.form["encoded_Embarked"])
        encoded_Sex = int(request.form["encoded_Sex"])

        # Create numpy input array
        test_values = np.array(
            [[Pclass, Age, SibSp, Parch, Fare, encoded_Embarked, encoded_Sex]]
        )

        prediction = str(load_model.predict(test_values)[0])
        
        if prediction == "1":
            prediction = "Survived" 
        else:
            prediction = "Did not survive"
            
    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)

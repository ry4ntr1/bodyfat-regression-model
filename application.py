from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            age=float(request.form.get("age")),
            weight=float(request.form.get("weight")),
            height=float(request.form.get("height")),
            neck=float(request.form.get("neck")),
            chest=float(request.form.get("chest")),
            abdomen=float(request.form.get("abdomen")),
            hip=float(request.form.get("hip")),
            thigh=float(request.form.get("thigh")),
            knee=float(request.form.get("knee")),
            ankle=float(request.form.get("ankle")),
            biceps=float(request.form.get("biceps")),
            forearm=float(request.form.get("forearm")),
            wrist=float(request.form.get("wrist"))
        )
        pred_df = data.get_data_as_dataframe()
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        return render_template("home.html", results=results[0])

if __name__ == "__main__":
    app.run(debug=True)

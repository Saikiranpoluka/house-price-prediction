from flask import Flask, request, render_template

from src.predict import predict_house_price


application = Flask(__name__)
app = application


@app.route("/")
def index():
    """
    Render the landing page.

    Returns:
        str: Rendered index.html template.
    """
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Handle house price prediction requests.

    For GET request:
        Display the prediction form.

    For POST request:
        Read user inputs from the form, send them to the prediction
        pipeline, and display the predicted house price.

    Returns:
        str: Rendered home.html template.
    """
    if request.method == "POST":
        try:
            input_data = {
                "longitude": float(request.form.get("longitude")),
                "latitude": float(request.form.get("latitude")),
                "housing_median_age": float(request.form.get("housing_median_age")),
                "total_rooms": float(request.form.get("total_rooms")),
                "total_bedrooms": float(request.form.get("total_bedrooms")),
                "population": float(request.form.get("population")),
                "households": float(request.form.get("households")),
                "median_income": float(request.form.get("median_income")),
                "ocean_proximity": request.form.get("ocean_proximity"),
            }

            prediction = predict_house_price(input_data)

            return render_template("home.html", result=prediction)

        except Exception:
            import traceback
            traceback.print_exc()

            return render_template(
                "home.html",
                error="Something went wrong. Please check your input values.",
            )

    return render_template("home.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
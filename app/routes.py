from app import app
from flask import render_template, request, flash, redirect
import requests
import base64
from app.classes import CLASS_NAMES

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
@app.route("/index", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]

        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            image_bytes = file.read()

            data = base64.b85encode(image_bytes).decode("utf-8")

            url = "https://4bfgt4ns83.execute-api.us-east-1.amazonaws.com/predict"

            response = requests.post(url, data=data)

            return CLASS_NAMES[response.json()]

    return render_template("index.html")

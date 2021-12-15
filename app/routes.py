from app import app
from flask import render_template, request, flash, redirect
from werkzeug.utils import secure_filename
import os
import requests
import base64
from app.classes import CLASS_NAMES
import timeit

UPLOAD_FOLDER = "."
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


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
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            with open(filepath, "rb") as image_source:
                image_bytes = image_source.read()

            data = base64.b85encode(image_bytes).decode("utf-8")

            url = "https://ohld3opc0h.execute-api.us-east-1.amazonaws.com/prod"

            start = timeit.default_timer()

            response = requests.post(
                url, data=data, headers={"Content-Type": "application/octet-stream"}
            )
            end = timeit.default_timer()
            total = end - start

            os.remove(filepath)

            return f"{CLASS_NAMES[response.json()]} {total}"

    return render_template("index.html")

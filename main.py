from flask import Flask, render_template, request, redirect

import assignment_code, data_description, data_exploration, data_modeling, data_preparation, index, model_deployment, model_evaluation

app = Flask(__name__)

@app.route("/")
def home ():
    return render_template("page.html", page='home', body=index.html())

@app.route("/data-description")
def data_description_page():
    return render_template("page.html", page='description', body=data_description.html())

@app.route("/data-exploration")
def data_exploration_page():
    return render_template("page.html", page='exploration', body=data_exploration.html())

@app.route("/data-preparation")
def data_preparation_page():
    return render_template("page.html", page='preparation', body=data_preparation.html())

@app.route("/data-modeling")
def data_modeling_page():
    return render_template("page.html", page='modeling',body=data_modeling.html())

@app.route("/model-evaluation")
def model_evaluation_page():
    return render_template("page.html", page='evaluation',body=model_evaluation.html())

@app.route("/model-deployment", methods=["GET", "POST"])
def model_deployment_pagge():
    data = None
    if request.method == 'POST':
        # Get the data from the POST request.
        data = request.form.to_dict()
    return render_template("page.html", page='deployment',body=model_deployment.html(form=data))

@app.route("/code")
def assignment_code_page():
    return render_template("page.html", page='code',body=assignment_code.html())
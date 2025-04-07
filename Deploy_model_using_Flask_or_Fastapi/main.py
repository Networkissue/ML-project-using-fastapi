#fastapi
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load your model
model = pickle.load(open('savedmodel.sav', 'rb'))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": ""})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    result = prediction[0]  # Already a string/class label
    return templates.TemplateResponse("index.html", {"request": request, "result": result})

# Only needed if you're running the file directly
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)



# ---------------------------------------
#flask
# from flask import Flask, render_template, request
# import pickle

# app = Flask(__name__)
# model = pickle.load(open('savedmodel.sav', 'rb'))


# @app.route('/')
# def home():
#     return render_template('index.html', result='')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # get form values
#     sepal_length = float(request.form['sepal_length'])
#     sepal_width = float(request.form['sepal_width'])
#     petal_length = float(request.form['petal_length'])
#     petal_width = float(request.form['petal_width'])

#     prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    
#     # optional: convert to species name
#     species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
#     # result = species[prediction[0]]
#     # result = species[int(prediction[0])]
#     # No need to map it to species, just return prediction[0] as it's already a string
#     result = prediction[0]



#     return render_template('index.html', result=result)


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080, debug=True)

from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('savedmodel.sav', 'rb'))


@app.route('/')
def home():
    return render_template('index.html', result='')

@app.route('/predict', methods=['POST'])
def predict():
    # get form values
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # optional: convert to species name
    species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    # result = species[prediction[0]]
    # result = species[int(prediction[0])]
    # No need to map it to species, just return prediction[0] as it's already a string
    result = prediction[0]



    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

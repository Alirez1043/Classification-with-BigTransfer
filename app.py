


import tensorflow as tf
from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

MODEL_PATH = "static/models/model-1"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

IMAGE_SIZE = 224

# Read the image from path and preprocess
def load_and_preprocess_image(img):
    image = tf.image.decode_image(img, expand_animations=False)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image /= 255.0  # normalize to [0,1] range
    return image


# Predict & classify image
def classify(model, image_path):

    image = load_and_preprocess_image(image_path)
    image = tf.expand_dims(image ,axis=0)
    preds = model.predict(image)

    class_names = ["Leopard" ,"Tiger" ,"Horse" ,"Lion" ,"Zebra"]
    id = np.array(preds).argmax()
    label = class_names[id]

    probas = np.round_(preds * 100 ,decimals = 2)

    answer = {
        'label': label,
        'confidence': {str(class_): str(proba) for class_ , proba in zip(class_names, probas.tolist()[0])}
    }

    return answer


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST", "GET"])
def predict():

    if request.method == "GET":
        return render_template("index.html")

    else:
        if request.files.get("image") :

            image = request.files.get("image").read()
            return jsonify(classify(model, image)) ,200
        else:
            answer = {"error" : "Your input file is not an image"}
            return jsonify(answer) ,400

@app.route("/statistic" ,methods = ["GET"])
def statistic():
    class_names = ["Leopard" ,"Tiger" ,"Horse" ,"Lion" ,"Zebra"]
    precision = "98.6%"
    recall = "98.6%"
    f1_score = "98.5%"
    conf_mat = {
                "A-Leopard" :"[198 ,2 ,0 ,0 ,0]",
                "B-Tiger":"[1 ,195 ,0 ,3 ,1]",
                "C-Horse":"[0 ,0 ,199 ,1 ,0]",
                "D-Lion":"[1 ,2 ,1 ,196 ,0]",
                "E-Zebra":"[0 ,1 ,1 ,0 ,198]"
                }
    answer = {
        "class_names":class_names,
        "Confusion_Matrix" : conf_mat ,
        "Precision":precision ,
        "Recall":recall ,
        "F1_Score":f1_score
    }

    return jsonify(answer) ,200

if __name__ == "__main__":
    app.debug = True
    app.run()

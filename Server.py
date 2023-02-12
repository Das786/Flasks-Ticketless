from flask import Flask,jsonify,request,json
from flask_cors import CORS, cross_origin
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
import tensorflow_hub as hub
from geopy.geocoders import Nominatim
import os
from werkzeug.utils import secure_filename
import pickle
import random
from datetime import datetime


model_url = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'
labels = 'landmarks_classifier_asia_V1_label_map.csv'
df = pd.read_csv(labels)
labels = dict(zip(df.id, df.name))

with open('config.json', 'r') as c:
    params = json.load(c)["params"]

app = Flask(__name__)
# CORS(app,support_credentials=True)
app.config['SECRET_KEY'] = 'asdgfghjk'
app.config['UPLOAD_FOLDER'] = params['upload_location']




@app.route("/location-recongition", methods=["GET", "POST"])
def locreco():
   if(request.method == 'POST'):
    f = request.files['file']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)))
    print("Uploaded Successfully")
    print(f.filename)
    image = "D://MajorProject//TicketlessEntry//Flasks//uploads//"+f.filename
    print(image)
    img_shape = (321, 321)
    classifier = tf.keras.Sequential(
        [hub.KerasLayer(model_url, input_shape=img_shape + (3,), output_key="predictions:logits")])
    img = PIL.Image.open(image)
    img = img.resize(img_shape)
    img1 = img
    img = np.array(img) / 255.0
    img = img[np.newaxis]
    result = classifier.predict(img)
    fresult = labels[np.argmax(result)],img1
    print("Prediction Location is ",fresult[0])
    geolocator = Nominatim(user_agent="Your_Name")
    location = geolocator.geocode(fresult[0])
    print(location.address,location.latitude, location.longitude) 
    return jsonify([{
        "name": fresult[0],
        "lat": location.latitude,
        "log": location.longitude,
        "location": location.address
    }])

@app.route("/crowd-prediction",methods=["GET","POST"])
def crowdPred():
    date1 = request.json['date1']
    date2 = request.json['date2']
    print("selected Date", date2)
    d1 = datetime.strptime(date1, "%d/%m/%Y")
    d2 = datetime.strptime(date2, "%d/%m/%Y")
    delta = d2 - d1
    date = delta.days
    isHoliday = random.randint(0,1)
    isWeekend = random.randint(0,1)
    SPlace = request.json['SPlace']

    print(date,isHoliday,isWeekend,SPlace)
    loaded_model = pickle.load(open('Random_Forest.sav', 'rb'))
    result = loaded_model.predict(np.array([[date,isHoliday,isWeekend,SPlace]]).tolist()).tolist()
    return jsonify({"Prediction": round(result[0])})

@app.route("/members")
def members():
    json_members = [
{
    "userId": 1,
    "id": "1",
    "name": "Dass",
    "email": "thapa@technical.com",
    "website": "https://www.youtube.com/thapatechnical",
    "mobile": "1234567899",
    "image": "https://images.pexels.com/photos/220453/pexels-photo-220453.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=750&w=1260",
    "description": "quia et suscipit\nsuscipit recusandae consequuntur expedita et cum\nreprehenderit molestiae ut ut quas totam\nnostrum rerum est autem sunt rem eveniet architecto"
},
{
    "userId": 2,
    "id": "2",
    "name": "vinod thapa",
    "email": "vinod@technical.com",
    "website": "https://www.youtube.com/thapatechnical",
    "mobile": "1234567899",
    "image": "https://images.pexels.com/photos/547593/pexels-photo-547593.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=750&w=1260",
    "description": "est rerum tempore vitae\nsequi sint nihil reprehenderit dolor beatae ea dolores neque\nfugiat blanditiis voluptate porro vel nihil molestiae ut reiciendis\nqui aperiam non debitis possimus qui neque nisi nulla"
},
]

    return jsonify(json_members)




if __name__ == "__main__":
    app.run(port=3000,debug=True)
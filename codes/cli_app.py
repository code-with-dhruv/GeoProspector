
import requests as r
import cv2, os
import numpy as np
from tensorflow.keras.models import load_model

def get_static_latlng(lat, lng):

    print("Getting static map of", lat, ",", lng)

    url = "https://maps.googleapis.com/maps/api/staticmap?center={},{}&zoom=12&size=225x225&maptype=satellite&key=AIzaSyBHYiF8XJBMiuxfXtk5NDzW4hxNJcEkPkY".format(
        str(lat), str(lng)
    )
    try:
        a = r.get(url)
        f = open("static.png", "wb")
        f.write(a.content)
        f.close()

        rgb = cv2.imread("static.png")
    except:
        print("B - 0000")
        rgb = np.zeros((600, 600, 3), np.uint8)

    print(rgb.shape)
    return rgb
    
def classify_lat_long(img_arr):
    
    global model
    class_mapping = {'BareLand': 0, 'Commercial': 1, 'DenseResidential': 2, 'Desert': 3, 'Farmland': 4, 'Forest': 5, 'Industrial': 6, 'Meadow': 7, 'MediumResidential': 8, 'Mountain': 9, 'River': 10, 'SparseResidential': 11}
    class_mapping_inv = dict(zip(class_mapping.values(), class_mapping.keys()))

    pred = model.predict(np.array([cv2.resize(img_arr, (225, 225))]))
    ind = list(pred[0]).index(max(list(pred[0])))
    return class_mapping_inv[ind]

def check_renewability(pred):
    if pred in ["Commercial", "Industrial", "DenseResidential", "MediumResidential", "SparseResidential", 
                "Mountain", "River", "Forest"]:
        return "Un-Suitable for building a Renewable Energy Plant"
    elif pred in ["BareLand", "Desert"]:
        return "Geo-Location Suitable for a Wind-Mill Plant"
    elif pred in ["Farmland", "Meadow"]:
        return "Geo-Location Suitable for a Solar Power Plant"

print("PROGRAM LOADING...")

print("Loading model")
model = load_model("model2.h5")
print("Done loading model")

while True:

    p = input("\nEnter the latitude and longitude of the Area of Interest: ").split(",")
    prompt =[i.strip() for i in p]
    if "quit" in prompt:
        break

    img_arr = get_static_latlng(17.542220, 78.449591)#(prompt[0], prompt[1])
    pred = classify_lat_long(img_arr)
    print("Geo-Location classified as", pred)
    print("Inference:", check_renewability(pred))

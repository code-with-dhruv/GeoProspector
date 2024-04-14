
import os, cv2, pickle, random
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow import keras
import numpy as np

#dataset Preparation

dataset_path = "./Dataset/AID/"

if "X.obj" not in os.listdir() or "y.obj" not in os.listdir():
    class_map = {}
    index = 0
    X = []
    y = []
    for class_name in os.listdir(dataset_path):

        print("In class:", class_name)
        if class_name not in class_map:
            class_map[class_name] = index

        ohe = [0.0 for i in range(12)]
        ohe[index] = 1.0
        index += 1

        for image_file in os.listdir( dataset_path + "/" + class_name):
            X.append(cv2.resize(cv2.imread(dataset_path + "/" + class_name + "/" + image_file), (225, 225)))
            y.append(ohe)

    X = np.array(X, dtype=np.int8)
    y = np.array(y, dtype=float)

    pickle.dump(X, open("X.obj", "wb"))
    pickle.dump(y, open("y.obj", "wb"))
else:
    X = pickle.load(open("X.obj", "rb"))
    y = pickle.load(open("y.obj", "rb"))

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3,
                                                    random_state=100)

if "model.h5" not in os.listdir():
    #model building

    model = Sequential(layers=VGG16(input_shape = (225, 225, 3), 
    include_top = False,
    weights = 'imagenet').layers)
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(25, activation="relu"))
    model.add(Dense(12, activation="softmax"))

    print(model.layers[0].input_shape)

else:
    model = keras.models.load_model("model.h5")

while True:

    if input("Train again?") != "y":
        break

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3,
                                                    random_state=random.randint(0, 1000))
    
    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.000001),
                metrics=["acc"])
    model.fit(train_x, train_y, epochs=5, batch_size=16, validation_split=0.2)
    model.save("model2.h5")

    model.evaluate(test_x, test_y)

model.evaluate(test_x, test_y)


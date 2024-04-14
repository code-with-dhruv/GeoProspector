
import os, cv2, random
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import numpy as np

dataset_specific = "dataset/Forest_seg/Forest Segmented/Forest Segmented/images"

X = []
y = []
y_label = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
for i in os.listdir(dataset_specific):
    X.append(cv2.resize(cv2.imread(dataset_specific + "/" + i), (225, 225)))
    y.append(y_label)

X = np.array(X, dtype=np.uint8)
y = np.array(y, dtype=float)

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=random.randint(0, 100))

model = load_model("model.h5")
model.compile(loss="categorical_crossentropy", optimizer="adam",
                metrics=["acc"])

model.fit(train_x[0:5], train_y[0:5], epochs=1, batch_size=4, validation_split=0.2)
model.save("model2.h5")
model.evaluate(test_x, test_y)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

model = keras.models.load_model('./model/best_model.hdf5')


def value_scaler(iw_val, if_val, vw_val, fp_val):
    X_input = np.array([[iw_val, if_val, vw_val, fp_val]])
    scaler = StandardScaler()
    scaler.scale_, scaler.mean_, scaler.var_ = (np.array([1.72448198, 5.27664937, 1.88255506, 22.33003242]),
                                                np.array([45.71929825, 141.26315789, 8.88596491, 79.03508772]),
                                                np.array([2.9738381, 27.84302862, 3.54401354, 498.6303478]))
    return scaler.transform(X_input).reshape(-1, 4)


def predict_welding(X_input):
    return model.predict(X_input)


if __name__ == "__main__":
    print(predict_welding(value_scaler([43, 146, 9.0, 60])))

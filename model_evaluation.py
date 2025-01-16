from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from data_visualization import load_data
import numpy as np

def evaluate_model(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test loss: {score[0]}")
    print(f"Test accuracy: {score[1]}")

def analyze_predictions(model, x_test, y_test):
    predictions = model.predict(x_test)
    for i in range(5):
        predicted_label = np.argmax(predictions[i])
        true_label = np.argmax(y_test[i])
        print(f"Bild {i}: Vorhergesagt: {predicted_label}, Tatsächlich: {true_label}")

if __name__ == "__main__":
    (_, _), (x_test, y_test) = load_data()
    x_test = x_test / 255.0
    y_test_cat = to_categorical(y_test, 10)
    model = load_model("fashion_mnist_model.h5")
    evaluate_model(model, x_test, y_test_cat)
    analyze_predictions(model, x_test, y_test_cat)

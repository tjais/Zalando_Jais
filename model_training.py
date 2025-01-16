from tensorflow.keras import Sequential, layers
from tensorflow.keras.utils import to_categorical
from data_visualization import load_data

def create_model():
    num_classes = 10  # Es gibt 10 Kleidungs-Kategorien

    model = Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax")  # Ausgabeschicht
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    return model

def train_model(model, x_train, y_train, batch_size=64, epochs=10, validation_split=0.1):
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split
    )
    return history

if __name__ == "__main__":
    (x_train, y_train), _ = load_data()

    x_train = x_train / 255.0
    y_train = to_categorical(y_train, 10)

    model = create_model()
    history = train_model(model, x_train, y_train)

    model.save("fashion_mnist_model.h5")
    print("Das optimierte Modell wurde gespeichert.")

# Beobachtungen zum Lernprozess
# Der Trainings-Loss sollte im Allgemeinen sinken, während die Trainings-Accuracy steigen sollte.
# Es ist wichtig, die Validierungs-Accuracy zu beobachten, da sie ein guter Indikator für Overfitting ist.
# Wenn die Validierungs-Accuracy nicht weiter steigt oder der Validierungs-Loss steigt, sollte man das Training frühzeitig abbrechen.

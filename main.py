from data_visualization import load_data, visualize_images, export_images_by_category, data_overview
from model_training import create_model, train_model
from model_evaluation import evaluate_model, analyze_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = load_data()
    data_overview(x_train, y_train)

    visualize_images(x_train, y_train)
    export_images_by_category(x_train, y_train)


    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)


    model = create_model()
    train_model(model, x_train, y_train_cat)

    model.save("fashion_mnist_model.h5")

    model = load_model("fashion_mnist_model.h5")
    evaluate_model(model, x_test, y_test_cat)

    analyze_predictions(model, x_test, y_test)

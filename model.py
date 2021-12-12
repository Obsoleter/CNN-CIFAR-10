import tensorflow.keras as keras
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


# Load data
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# Scale data
X_train = X_train / 255
X_test = X_test / 255


# CNN
network = keras.Sequential([
    # Input
    keras.layers.Input(shape=(32, 32, 3)),

    # Convolution layers
    keras.layers.Convolution2D(filters=24, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    keras.layers.Convolution2D(filters=48, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Dense layers
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

network.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

network.fit(X_train, y_train, epochs=10)


# Evaluation
accuracy = network.evaluate(X_test, y_test)[1]


# Confusion matrix
y_pred = network.predict(X_test).argmax(1)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title(f"Accuracy: {accuracy * 100}%", fontdict={'fontweight':'bold'})
plt.show()
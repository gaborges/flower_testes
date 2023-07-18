import os

import flwr as fl
import tensorflow as tf


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load model and data (MobileNetV2, CIFAR-10)
model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        #print(f"[Client {self}] config")
        #print(model.get_weights())
        print("testeeee")
		
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
		
        model.set_weights(parameters)
        # Evaluate global model parameters on the local test data
        loss, accuracy = model.evaluate(x_test, y_test)
        #loss, accuracy = model.evaluate(x_test, y_test)
        print("test metrics: ",accuracy)
		
		# Return results, including the custom accuracy metric
        num_examples_test = len(x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=CifarClient())

import larq as lq
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
import time
import pandas as pd

# Common options for quantized layers
kwargs = dict(input_quantizer="ste_sign",
              kernel_quantizer="ste_sign",
              kernel_constraint="weight_clip")

def build_model():
    model = Sequential()

    # Input layer
    model.add(Dense(64, use_bias=False, input_shape=(784,)))
    model.add(BatchNormalization(scale=False))

    # Hidden layers
    model.add(Dense(64, use_bias=False))
    model.add(BatchNormalization(scale=False))

    model.add(Dense(32, use_bias=False))
    model.add(BatchNormalization(scale=False))

    # Output layer
    model.add(Dense(10, use_bias=False))
    model.add(BatchNormalization(scale=False))
    model.add(Activation("softmax"))

    return model

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# Prepare datasets
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)


# Learning rate scheduler
def scheduler(epoch, lr):
    return lr * 0.7 if epoch % 1 == 0 else lr

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)


# List of optimizers to try
optimizers = {
    'adam': tf.keras.optimizers.Adam(learning_rate=1.0),
    'adamax': tf.keras.optimizers.Adamax(learning_rate=1.0),
    'adadelta': tf.keras.optimizers.Adadelta(learning_rate=1.0)
}

# Placeholder to store results
results = []

for optimizer_name, optimizer in optimizers.items():
    model = build_model()
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'],
    )
    
    print(f'**** {optimizer_name} ****')
    for epoch in range(10):
        print(f'Epoch: {epoch}')
        
        start_time = time.time()
        
        # Train the model
        history = model.fit(
            train_dataset,
            epochs=1,
            validation_data=test_dataset,
            callbacks=[callback]
        )
        
        train_time = time.time() - start_time
        train_loss = history.history['loss'][0]
        train_accuracy = history.history['accuracy'][0]
        
        start_time = time.time()
        
        # Evaluate the model
        test_loss, test_acc = model.evaluate(test_dataset)
        print(f"Test accuracy: {test_acc:.4f}")
        
        test_time = time.time() - start_time
        
        # Save results
        results.append({
            "model_name": "Dense_MNIST",
            "optimizer_name": optimizer_name,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_accuracy,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "epoch_train_time": train_time,
            "epoch_test_time": test_time
        })

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv('./mnist_larq/training_results_mlp_lr.csv', index=False)

print("Training results saved to 'training_results.csv'")

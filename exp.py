import tensorflow as tf
import matplotlib.pyplot as plt
import time

print("Initializing training...")

# Define cluster specification
cluster_spec = {
    "ps": ["ns31:2222"], # ps = parameter server
    "worker": ["inv01:2222", "inv02:2222"]
}

# Strategy for Data Parallelism
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Data Parallelism Training
print("Starting Data Parallelism Training...")
with strategy.scope():
    model_data_parallel = create_model()
    start_time = time.time()
    model_data_parallel.fit(x_train, y_train, epochs=5)
    end_time = time.time()
data_parallelism_time = end_time - start_time
data_parallelism_accuracy = model_data_parallel.evaluate(x_test, y_test)[1]
print(f"Data Parallelism Training completed in {data_parallelism_time:.2f} seconds.")

# Model Parallelism Training
# Splitting the model into two parts and placing each on a different worker
print("Starting Model Parallelism Training...")
with tf.device("/job:worker/task:0"):
    input_layer = tf.keras.layers.Input(shape=(28, 28))
    flatten = tf.keras.layers.Flatten()(input_layer)
    dense1 = tf.keras.layers.Dense(64, activation='relu')(flatten)

with tf.device("/job:worker/task:1"):
    dense2 = tf.keras.layers.Dense(64, activation='relu')(dense1)
    dropout = tf.keras.layers.Dropout(0.2)(dense2)
    output_layer = tf.keras.layers.Dense(10)(dropout)

model_model_parallel = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
model_model_parallel.compile(optimizer='adam',
                             loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                             metrics=['accuracy'])

start_time = time.time()
model_model_parallel.fit(x_train, y_train, epochs=5)
end_time = time.time()
model_parallelism_time = end_time - start_time
model_parallelism_accuracy = model_model_parallel.evaluate(x_test, y_test)[1]
print(f"Model Parallelism Training completed in {model_parallelism_time:.2f} seconds.")

# Plotting function
def plot_results(times, accuracies):
    labels = ['Data Parallelism', 'Model Parallelism']
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot training times
    axs[0].bar(labels, times, color=['blue', 'green'])
    axs[0].set_ylabel('Training Time (s)')
    axs[0].set_title('Comparison of Data and Model Parallelism - Training Time')
    
    # Plot accuracies
    axs[1].bar(labels, accuracies, color=['blue', 'green'])
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Comparison of Data and Model Parallelism - Accuracy')
    
    plt.tight_layout()
    plt.savefig('comparison_graph.png')

# Plot the results
plot_results([data_parallelism_time, model_parallelism_time], 
             [data_parallelism_accuracy, model_parallelism_accuracy])
print("Training completed. Results saved to comparison_graph.png.")
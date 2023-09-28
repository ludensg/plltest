import tensorflow as tf

# Set TensorFlow configuration
config = tf.compat.v1.ConfigProto()
config.inter_op_parallelism_threads = 10
config.intra_op_parallelism_threads = 10
tf.compat.v1.Session(config=config)

print("Starting Parameter Server...")

cluster_spec = {
    "ps": ["ns31:2222"],
    "worker": ["inv01:2222", "inv02:2222"]
}

server = tf.distribute.Server(cluster_spec, job_name="ps", task_index=0)
print("Parameter Server started on ns31:2222")
server.join()

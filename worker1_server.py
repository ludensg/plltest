import tensorflow as tf

print("Starting Worker 1...")

cluster_spec = {
    "ps": ["ns31:2222"],
    "worker": ["inv01:2222", "inv02:2222"]
}

server = tf.distribute.Server(cluster_spec, job_name="worker", task_index=1)
print("Worker 1 started on inv02:2222")
server.join()

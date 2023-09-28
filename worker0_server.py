import tensorflow as tf

print("Starting Worker 0...")

cluster_spec = {
    "ps": ["ns31:2222"],
    "worker": ["inv01:2222", "inv02:2222"]
}

server = tf.distribute.Server(cluster_spec, job_name="worker", task_index=0)
print("Worker 0 started on inv01:2222")
server.join()

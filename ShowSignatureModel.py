import tensorflow as tf

# Učitaj model
model = tf.saved_model.load("saved_model/tourism_model")

# Prikaži sve operacije u grafu
for op in model.signatures["serving_default"].structured_outputs.keys():
    print(op)

# Prikaži sve operacije u modelu
for op in model.signatures["serving_default"].graph.get_operations():
    print(op.name)
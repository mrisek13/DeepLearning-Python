from keras.src.saving import load_model

# Učitavanje modela
model = load_model("tourism_model.keras")

# Izvoz u SavedModel format
model.export("saved_model/tourism_model")

print("Model uspješno spremljen u SavedModel formatu!")
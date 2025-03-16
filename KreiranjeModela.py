import tensorflow as tf
import keras
from keras import layers, Sequential
import pandas as pd
import numpy as np
from keras.src.layers import Dense
from keras.src.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

# from tensorflow.python import keras


data = {
    "Spol": ["Muški", "Muški", "Muški", "Muški", "Ženski", "Ženski", "Ženski", "Ženski"],
    "Regija": ["Zagreb", "Zagreb", "Varaždin", "Varaždin", "Zagreb", "Zagreb", "Varaždin", "Osijek"],
    "Starost": ["20-30", "30-40", "20-30", "30-40", "20-30", "30-40", "20-30", "30-40"],
    "Destinacija": ["Pag", "Cres", "Pag", "Dubrovnik", "Pag", "Dubrovnik", "Pag", "Cres"],
    "Smještaj": ["Apartman", "Hotel", "Apartman", "Hotel", "Kamp", "Hotel", "Kamp", "Hotel"]
}

df = pd.DataFrame(data)

# 📌 2️⃣ Enkodiranje ulaznih podataka (One-hot encoding za Spol, Regija, Starost)
encoder = OneHotEncoder()
X = encoder.fit_transform(df[["Spol", "Regija", "Starost"]]).toarray()

# 📌 3️⃣ Enkodiranje izlaznih podataka (Destinacija, Smještaj) u one-hot format
label_encoder_dest = LabelEncoder()
label_encoder_smjestaj = LabelEncoder()

y_dest = label_encoder_dest.fit_transform(df["Destinacija"])
y_smjestaj = label_encoder_smjestaj.fit_transform(df["Smještaj"])

# Pretvaranje u one-hot encoding
y_dest_encoded = to_categorical(y_dest)
y_smjestaj_encoded = to_categorical(y_smjestaj)

# Spajanje destinacije i smještaja u jedan izlazni vektor
y = np.hstack((y_dest_encoded, y_smjestaj_encoded))

# 📌 4️⃣ Podjela podataka na trenirajući i testni skup
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 5️⃣ Definicija neuronske mreže
model = Sequential([
    Dense(16, activation="relu", input_shape=(X.shape[1],)),
    Dense(8, activation="relu"),
    Dense(y.shape[1], activation="softmax")  # Broj izlaznih klasa
])

# 📌 6️⃣ Kompajliranje modela s categorical_crossentropy
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 📌 7️⃣ Treniranje modela
model.fit(X_train, y_train, epochs=50, batch_size=2, validation_data=(X_test, y_test))

# 📌 8️⃣ Spremanje modela
model.save("tourism_model.keras")

print("Model uspješno treniran i spremljen!")

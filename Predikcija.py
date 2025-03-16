import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd

from KreiranjeModela import df, model

encoder = OneHotEncoder()
X_sample_data = [["Muški", "Zagreb", "20-30"]]
#X_sample_data = [["Muški", "Zagreb", "30-40"]]
#X_sample_data = [["Muški", "Varaždin", "20-30"]]
#X_sample_data = [["Muški", "Varaždin", "30-40"]]
#X_sample_data = [["Ženski", "Zagreb", "20-30"]]
#X_sample_data = [["Ženski", "Zagreb", "30-40"]]
#X_sample_data = [["Ženski", "Varaždin", "20-30"]]
#X_sample_data = [["Ženski", "Osijek", "30-40"]]

#X_sample_data = [["Muški", "Osijek", "20-30"]]

# Moramo ponovno trenirati enkoder da zna sve moguće kategorije
df_sample = pd.DataFrame({
    "Spol": ["Muški", "Muški", "Muški", "Muški", "Ženski", "Ženski", "Ženski", "Ženski"],
    "Regija": ["Zagreb", "Zagreb", "Varaždin", "Varaždin", "Zagreb", "Zagreb", "Varaždin", "Osijek"],
    "Starost": ["20-30", "30-40", "20-30", "30-40", "20-30", "30-40", "20-30", "30-40"]
})

encoder.fit(df_sample)  # Moramo ponovno trenirati encoder da zna sve moguće vrijednosti
X_transformed = encoder.transform(X_sample_data).toarray()

# Predikcija
predikcija = model.predict(X_transformed)

# Odvajanje predikcija na destinaciju i smještaj
num_destinacija = len(set(df["Destinacija"]))  # Broj kategorija destinacija
predikcija_destinacija = predikcija[:, :num_destinacija]
predikcija_smjestaj = predikcija[:, num_destinacija:]

# Dekodiranje rezultata
label_encoder_dest = LabelEncoder()
label_encoder_smjestaj = LabelEncoder()

label_encoder_dest.fit(df["Destinacija"])
label_encoder_smjestaj.fit(df["Smještaj"])

destinacija_predicted = label_encoder_dest.inverse_transform([np.argmax(predikcija_destinacija)])
smjestaj_predicted = label_encoder_smjestaj.inverse_transform([np.argmax(predikcija_smjestaj)])

# Ispis rezultata
print(f"Predviđena destinacija: {destinacija_predicted[0]}")
print(f"Predviđena vrsta smještaja: {smjestaj_predicted[0]}")
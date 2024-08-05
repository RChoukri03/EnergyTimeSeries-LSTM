# Image 1
from keras.callbacks import Callback
import time
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, CSVLogger, Callback
import matplotlib.pyplot as plt
import pandas as pd
import time
import joblib
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0 = all logs, 1 = filter out INFO, 2 = filter out WARNING, 3 = filter out ERROR
import warnings
warnings.filterwarnings('ignore')

# Vérifiez si un GPU est disponible et configurez TensorFlow pour l'utiliser
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print("GPU trouvé, utilisé pour l'entraînement.")
    except RuntimeError as e:
        # Les erreurs se produisent lorsque la mémoire GPU est insuffisante
        print(e)
else:
    print("GPU non trouvé, utilisation du CPU à la place.")


def adjust_and_save_scaler(dataset_name, train_path, target_path, model_path):
    all_values = []

    train_files = sorted(os.listdir(os.path.join(train_path, dataset_name)))
    target_files = sorted(os.listdir(os.path.join(target_path, dataset_name)))

    for train_file, target_file in zip(train_files, target_files):
        train_sequence = pd.read_csv(os.path.join(train_path, dataset_name, train_file))
        target_sequence = pd.read_csv(os.path.join(target_path, dataset_name, target_file))
        all_values.extend(train_sequence["Valeur"].values)
        all_values.extend(target_sequence["Valeur"].values)

    scaler = MinMaxScaler()
    scaler.fit(np.array(all_values).reshape(-1, 1))

    # Enregistrer le scaler
    joblib.dump(scaler, os.path.join(model_path, dataset_name, 'scaler.gz'))




class CustomCallback(Callback):
    def __init__(self, i, path):
        super().__init__()
        self.i = i
        self.path = path
        self.start_time = time.time()
        self.best_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('loss')
        # Sauvegarde du meilleur modèle à la fin de chaque époque
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.model.save(f'{self.path}/best_model{self.i}.h5')

    def on_train_end(self, logs=None):
        # Sauvegarde du modèle final et affichage du temps total d'entraînement
        self.model.save(f'{self.path}/final_model{self.i}.h5')
        total_time = time.time() - self.start_time
        print(f"Total training time: {total_time:.2f}s")



INPUT_SEQ_LENGTH = 2*30*24*6  # Calculé comme 2 mois de données à une fréquence de 6 pas par heure
OUTPUT_SEQ_LENGTH = 1*30*24*6  # Calculé comme 1 mois de données à une fréquence de 6 pas par heure


def data_loader(dataset_name, train_path, target_path, batch_size, model_path):
    # Charger le scaler enregistré
    scaler = joblib.load(os.path.join(model_path, dataset_name, 'scaler.gz'))

    while True:
        X_batch, y_batch = [], []

        train_files = sorted(os.listdir(os.path.join(train_path, dataset_name)))
        target_files = sorted(os.listdir(os.path.join(target_path, dataset_name)))

        for train_file, target_file in zip(train_files, target_files):
            train_sequence = pd.read_csv(os.path.join(train_path, dataset_name, train_file))
            target_sequence = pd.read_csv(os.path.join(target_path, dataset_name, target_file))
            train_sequence_scaled = scaler.transform(train_sequence["Valeur"].values.reshape(-1, 1))
            target_sequence_scaled = scaler.transform(target_sequence["Valeur"].values.reshape(-1, 1))

            X_batch.append(train_sequence_scaled.reshape(INPUT_SEQ_LENGTH, 1))
            y_batch.append(target_sequence_scaled.reshape(OUTPUT_SEQ_LENGTH))

            if len(X_batch) == batch_size:
                yield np.array(X_batch), np.array(y_batch)
                X_batch, y_batch = [], []





# Définir les longueurs de séquence d'entrée et de sortie
INPUT_SEQ_LENGTH = 2*30*24*6  # Calculé comme 2 mois de données à une fréquence de 6 pas par heure
OUTPUT_SEQ_LENGTH = 1*30*24*6  # Calculé comme 1 mois de données à une fréquence de 6 pas par heure

# Initialisation du modèle séquentiel
model = Sequential()

# Ajout de la première couche LSTM
# 512 unités, retourne la séquence complète pour permettre l'empilement de LSTM
model.add(LSTM(512, input_shape=(INPUT_SEQ_LENGTH, 1), return_sequences=True))
# Ajout d'une couche de Dropout pour réduire le surajustement
model.add(Dropout(0.2))

# Ajout de la deuxième couche LSTM
# 256 unités, retourne uniquement la dernière sortie de la séquence
model.add(LSTM(256))
# Ajout d'une dernière couche de Dropout pour réduire davantage le surajustement
model.add(Dropout(0.1))

# Ajout d'une couche Dense pour la sortie
# La taille de sortie est égale à OUTPUT_SEQ_LENGTH, activation linéaire pour la régression
model.add(Dense(OUTPUT_SEQ_LENGTH, activation='linear'))

# Compilation du modèle
# Utilisation de la fonction de perte 'mean squared error' pour une tâche de régression
# Optimiseur Adam pour l'ajustement efficace des poids
model.compile(loss="mse", optimizer=Adam())

# Configuration des chemins et des paramètres de l'entraînement
train_path = 'train/'
target_path = 'target/'
model_path = 'model/'
dataset_name = 'bureau/'

# Exemple d'appel de la fonction
adjust_and_save_scaler(dataset_name, train_path, target_path, model_path)
# Initialisation des callbacks
# CSVLogger pour enregistrer l'historique de la perte dans un fichier CSV
csv_logger = CSVLogger(f'{model_path}/{dataset_name}/loss_history.csv', append=True, separator=';')

# EarlyStopping pour arrêter l'entraînement si la perte ne s'améliore plus
early_stopping = EarlyStopping(monitor='loss', patience=300, verbose=1)

# CustomCallback pour les actions personnalisées en fin d'époque et d'entraînement
custom_callback = CustomCallback(i=0, path=f'{model_path}/{dataset_name}')


BATCH_SIZE = 9  # Utilisation de tous les exemples en un seul batch

# Ajustement de steps_per_epoch
steps_per_epoch = 1  # Un seul batch par époque


# Exemple d'appel de la fonction
train_generator = data_loader(dataset_name, train_path, target_path, BATCH_SIZE, model_path)
# # Chargement des données
# train_generator = data_loader(dataset_name, train_path, target_path)  # Assurez-vous que data_loader est défini ailleurs dans votre code
# steps_per_epoch = 9   # Ce devrait être basé sur votre taille de données / taille du batch

# Hyperparamètres de l'entraînement
epochs = 2000
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print("GPU trouvé, utilisé pour l'entraînement.")
    except RuntimeError as e:
        # Les erreurs se produisent lorsque la mémoire GPU est insuffisante
        print(e)
else:
    print("GPU non trouvé, utilisation du CPU à la place.")
# Lancer l'entraînement
history = model.fit(
    train_generator,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    callbacks=[custom_callback, csv_logger, early_stopping]
)

# Tracer la courbe de la perte après l'entraînement
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'{model_path}/{dataset_name}/loss_curve.png')
plt.show()
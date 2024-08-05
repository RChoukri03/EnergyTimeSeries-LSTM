import os
import gc  # Garbage Collector
import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, CSVLogger, Callback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Verify if a GPU is available and configure TensorFlow to use it
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logging.info("GPU found, using for training.")
    except RuntimeError as e:
        logging.error(f"Error setting GPU: {e}")
else:
    logging.info("No GPU found, using CPU instead.")

INPUT_SEQ_LENGTH = 2 * 30 * 24 * 6  # 2 months of data at 6 steps per hour
OUTPUT_SEQ_LENGTH = 1 * 30 * 24 * 6  # 1 month of data at 6 steps per hour

class CustomCallback(Callback):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.startTime = time.time()
        self.bestLoss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        currentLoss = logs.get('loss')
        if currentLoss < self.bestLoss:
            self.bestLoss = currentLoss
            self.model.save(f'{self.path}/best_model.h5')

    def on_train_end(self, logs=None):
        self.model.save(f'{self.path}/final_model.h5')
        totalTime = time.time() - self.startTime
        logging.info(f"Total training time: {totalTime:.2f}s")

def adjustAndSaveScaler(datasetName, trainPath, targetPath, modelPath):
    allValues = []
    trainFiles = sorted(os.listdir(os.path.join(trainPath, datasetName)))
    targetFiles = sorted(os.listdir(os.path.join(targetPath, datasetName)))
    for trainFile, targetFile in zip(trainFiles, targetFiles):
        trainSequence = pd.read_csv(os.path.join(trainPath, datasetName, trainFile))
        targetSequence = pd.read_csv(os.path.join(targetPath, datasetName, targetFile))
        allValues.extend(trainSequence["Valeur"].values)
        allValues.extend(targetSequence["Valeur"].values)
    scaler = MinMaxScaler()
    scaler.fit(np.array(allValues).reshape(-1, 1))
    joblib.dump(scaler, os.path.join(modelPath, datasetName, 'scaler.gz'))

def dataLoader(datasetName, trainPath, targetPath, batchSize, modelPath):
    scaler = joblib.load(os.path.join(modelPath, datasetName, 'scaler.gz'))
    while True:
        XBatch, yBatch = [], []
        trainFiles = sorted(os.listdir(os.path.join(trainPath, datasetName)))
        targetFiles = sorted(os.listdir(os.path.join(targetPath, datasetName)))
        for trainFile, targetFile in zip(trainFiles, targetFiles):
            trainSequence = pd.read_csv(os.path.join(trainPath, datasetName, trainFile))
            targetSequence = pd.read_csv(os.path.join(targetPath, datasetName, targetFile))
            trainSequenceScaled = scaler.transform(trainSequence["Valeur"].values.reshape(-1, 1))
            targetSequenceScaled = scaler.transform(targetSequence["Valeur"].values.reshape(-1, 1))
            XBatch.append(trainSequenceScaled.reshape(INPUT_SEQ_LENGTH, 1))
            yBatch.append(targetSequenceScaled.reshape(OUTPUT_SEQ_LENGTH))
            if len(XBatch) == batchSize:
                yield np.array(XBatch), np.array(yBatch)
                XBatch, yBatch = [], []

def buildAndTrainModel(config):
    trainPath = config['train_path']
    targetPath = config['target_path']
    modelPath = config['model_path']
    datasetName = config['dataset_name']
    batchSize = config['batch_size']
    epochs = config['epochs']

    adjustAndSaveScaler(datasetName, trainPath, targetPath, modelPath)

    model = Sequential()
    model.add(LSTM(512, input_shape=(INPUT_SEQ_LENGTH, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.1))
    model.add(Dense(OUTPUT_SEQ_LENGTH, activation='linear'))
    model.compile(loss="mse", optimizer=Adam())

    csvLogger = CSVLogger(f'{modelPath}/{datasetName}/loss_history.csv', append=True, separator=';')
    earlyStopping = EarlyStopping(monitor='loss', patience=300, verbose=1)
    customCallback = CustomCallback(path=f'{modelPath}/{datasetName}')

    trainGenerator = dataLoader(datasetName, trainPath, targetPath, batchSize, modelPath)

    stepsPerEpoch = 1

    history = model.fit(
        trainGenerator,
        epochs=epochs,
        steps_per_epoch=stepsPerEpoch,
        callbacks=[customCallback, csvLogger, earlyStopping]
    )

    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{modelPath}/{datasetName}/loss_curve.png')
    plt.show()

if __name__ == "__main__":
    datasets = ['hospital', 'office', 'mixed']
    for dataset in datasets:
        logging.info(f"Training model for {dataset} dataset")
        buildAndTrainModel(config[dataset])

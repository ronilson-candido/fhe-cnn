import matplotlib.pyplot as plt
import psutil
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import to_categorical
import time


def monitor_resources():
    process = psutil.Process()
    cpu_usage = process.cpu_percent(interval=1)
    ram_usage = process.memory_info().rss / (1024 * 1024)  
    return cpu_usage, ram_usage


def plot_accuracy_loss(accuracy, loss, epochs):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), accuracy, label='Acurácia', color='blue', marker='o')
    plt.title('Acurácia ao Longo das Iterações')
    plt.xlabel('Epoch')
    plt.ylabel('Acurácia')
    plt.grid(True)
    

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), loss, label='Loss', color='red', marker='o')
    plt.title('Loss ao Longo das Iterações')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_cpu_ram(cpu_usage, ram_usage, epochs):
    plt.figure(figsize=(12, 6))
    

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), cpu_usage, label='Uso de CPU', color='green', marker='o')
    plt.title('Uso de CPU ao Longo das Iterações')
    plt.xlabel('Epoch')
    plt.ylabel('Uso de CPU (%)')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), ram_usage, label='Consumo de RAM', color='purple', marker='o')
    plt.title('Consumo de RAM ao Longo das Iterações')
    plt.xlabel('Epoch')
    plt.ylabel('Consumo de RAM (MB)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def train_and_monitor():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255
    x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = Sequential()
    model.add(Dense(512, input_shape=(28 * 28,), activation='relu'))
    model.add(Dense(10, activation='softmax'))
    

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    

    epochs = 10


    accuracy = []
    loss = []
    cpu_usage = []
    ram_usage = []

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1, batch_size=128, verbose=1)
        

        accuracy.append(history.history['val_accuracy'][0])
        loss.append(history.history['val_loss'][0])
        

        cpu, ram = monitor_resources()
        cpu_usage.append(cpu)
        ram_usage.append(ram)
        
        time.sleep(1) 

    plot_accuracy_loss(accuracy, loss, epochs)
    plot_cpu_ram(cpu_usage, ram_usage, epochs)

train_and_monitor()

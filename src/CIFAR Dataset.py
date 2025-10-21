import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model

class CIFAR10Data:
  def __init__(self):
    self.load_data()

  def load_data(self):
    cifar10 = tf.keras.datasets.cifar10
    (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
    print('Train/Test shapes:', self.x_train.shape, self.y_train.shape, self.x_test.shape, self.y_test.shape)
    self.x_train = self.x_train / 255.0
    self.x_test = self.x_test / 255.0
    self.y_train = self.y_train.flatten()
    self.y_test = self.y_test.flatten()
    self.num_classes = len(set(self.y_train))
    print('Number of classes:', self.num_classes)

  def plot_samples(self, num=25):
    fig, ax = plt.subplots(5, 5)
    k = 0
    for i in range(5):
      for j in range(5):
        ax[i][j].imshow(self.x_train[k], aspect='auto')
        ax[i][j].axis('off')
        k += 1
    plt.suptitle('Sample CIFAR-10 Images')
    plt.show()

class CIFAR10Model:
  def __init__(self, input_shape, num_classes):
    self.model = self.build_model(input_shape, num_classes)

  def build_model(self, input_shape, num_classes):
    i = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(i, x)
    return model

  def compile(self):
    self.model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

  def summary(self):
    self.model.summary()

  def train(self, x_train, y_train, x_test, y_test, epochs=50, batch_size=32, augment=False):
    if not augment:
      history = self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)
    else:
      data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
      train_generator = data_generator.flow(x_train, y_train, batch_size)
      steps_per_epoch = x_train.shape[0] // batch_size
      history = self.model.fit(train_generator, validation_data=(x_test, y_test),
                  steps_per_epoch=steps_per_epoch, epochs=epochs)
    return history

  def save(self, path='cifar10_model.h5'):
    self.model.save(path)
    print(f'Model saved to {path}')

class CIFAR10Evaluator:
  def __init__(self, labels=None):
    if labels is None:
      self.labels = 'airplane automobile bird cat deer dog frog horse ship truck'.split()
    else:
      self.labels = labels

  def plot_accuracy(self, history):
    plt.plot(history.history['accuracy'], label='acc', color='red')
    plt.plot(history.history['val_accuracy'], label='val_acc', color='green')
    plt.legend()
    plt.title('Training Accuracy')
    plt.show()

  def predict_and_display(self, model, x_test, y_test, image_number=0):
    plt.imshow(x_test[image_number])
    plt.title('Test Image')
    plt.axis('off')
    plt.show()
    p = np.expand_dims(x_test[image_number], axis=0)
    pred = model.predict(p)
    predicted_label = self.labels[int(np.argmax(pred, axis=1)[0])]
    original_label = self.labels[int(y_test[image_number])]
    print(f'Original label is {original_label} and predicted label is {predicted_label}')

# Example usage
if __name__ == '__main__':
  print('TensorFlow version:', tf.__version__)
  # Load and preprocess data
  data = CIFAR10Data()
  data.plot_samples()

  # Build and train model
  model = CIFAR10Model(input_shape=data.x_train[0].shape, num_classes=data.num_classes)
  model.compile()
  model.summary()
  history = model.train(data.x_train, data.y_train, data.x_test, data.y_test, epochs=10, batch_size=32, augment=True)
  model.save('cifar10_model.h5')

  # Evaluate and predict
  evaluator = CIFAR10Evaluator()
  evaluator.plot_accuracy(history)
  evaluator.predict_and_display(model.model, data.x_test, data.y_test, image_number=0)

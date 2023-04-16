from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# Classe per l'attacco LinfPGD
class LinfPGDAttack:
  def __init__(self, model, epsilon, k, a, random_start, loss_func):
    """Inizializzazione dei parametri di attacco. L'attacco esegue k passi
       di dimensione a, rimanendo sempre entro epsilon dal punto iniziale."""
    self.model = model
    self.epsilon = epsilon
    self.k = k
    self.a = a
    self.rand = random_start

    if loss_func == 'xent':
      self.loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    elif loss_func == 'cw':
      label_mask = tf.one_hot(model.y_input,
                              10,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax
                                  - 1e4*label_mask, axis=1)
      self.loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
    else:
      print('Funzione di perdita sconosciuta. Si utilizza la cross-entropy per default')
      self.loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
      
  
  
  def perturb(self, x_nat, y):
    """Dato un set di esempi (x_nat, y), restituisce un set di esempi avversari
       entro epsilon di x_nat in norma l_infinity."""
    if self.rand:
      x = tf.Variable(x_nat) + tf.cast(tf.random.uniform(x_nat.shape, -self.epsilon, self.epsilon), tf.float32)
      x = tf.clip_by_value(x, 0, 1) # assicura un range di pixel valido
    else:
      x = tf.Variable(x_nat)

    for i in range(self.k):
      with tf.GradientTape() as tape:
        tape.watch(x)
        loss = self.loss(y, self.model(x))
      grad = tape.gradient(loss, x)

      x += self.a * tf.sign(grad)

      x = tf.clip_by_value(x, x_nat - self.epsilon, x_nat + self.epsilon) 
      x = tf.clip_by_value(x, 0, 1) # assicura un range di pixel valido

    return x



if __name__ == '__main__':
  import json
  import sys
  import math

  
  from model import Model

  # Caricamento delle impostazioni di configurazione
  with open('config.json') as config_file:
    config = json.load(config_file)


  model_file = tf.train.latest_checkpoint(config['model_dir'])
  if model_file is None:
    print('No model found')
    sys.exit()


  # Caricamento del modello salvato
  model = Model()
  attack = LinfPGDAttack(model,
                         config['epsilon'],
                         config['k'],
                         config['a'],
                         config['random_start'],
                         config['loss_func'])
  saver = tf.keras.Model.save_weights(model,'savers/my_model_weights')
  


# Caricamento del dataset MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train.astype(np.float32) / 255.0, axis=-1)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
x_test = np.expand_dims(x_test.astype(np.float32) / 255.0, axis=-1)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Costruzione del modello
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilazione del modello
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Addestramento del modello
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Valutazione del modello sul test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Test accuracy:', test_acc)

# Attacco di LinfPGD sul test set
attack = LinfPGDAttack(model=model, epsilon=0.1, k=40, a=0.01, random_start=True, loss_func='ce')
adv_x_test = attack.perturb(x_test, y_test)

# Valutazione del modello sull'insieme di test avversario
test_loss_adv, test_acc_adv = model.evaluate(adv_x_test, y_test, verbose=2)
print('Test accuracy on adversarial examples:', test_acc_adv)


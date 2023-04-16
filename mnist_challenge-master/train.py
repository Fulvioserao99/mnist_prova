import tensorflow as tf
from datetime import datetime
import json
import os
import shutil
import numpy as np
from timeit import default_timer as timer
from tensorflow.keras.datasets import mnist
from model import Model
from pgd_attack import LinfPGDAttack

with open('config.json') as config_file:
    config = json.load(config_file)

# Setting up training parameters
tf.random.set_seed(config['random_seed'])
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
batch_size = config['training_batch_size']

# Setting up the data and the model
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
mnist_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
model = Model()

# Setting up the optimizer
optimizer = tf.optimizers.Adam(1e-4)

@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        # Compute Adversarial Perturbations
        x_batch_adv = attack.perturb(x_batch, y_batch)
        nat_dict = {model.x_input: x_batch, model.y_input: y_batch}
        adv_dict = {model.x_input: x_batch_adv, model.y_input: y_batch}
        # Forward pass for natural images
        nat_logits = model(nat_dict['x_input'], training=True)
        nat_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(nat_dict['y_input'], nat_logits, from_logits=True))
        # Forward pass for adversarial images
        adv_logits = model(adv_dict['x_input'], training=True)
        adv_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(adv_dict['y_input'], adv_logits, from_logits=True))
        # Total loss
        loss = nat_loss + adv_loss
    # Compute gradients and update weights
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # Update global step
    global_step.assign_add(1)

# Set up adversary
attack = LinfPGDAttack(model,
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

# We add accuracy and loss twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
test_loss_metric = tf.keras.metrics.Mean(name='test_loss')

train_summary_writer = tf.summary.create_file_writer(os.path.join(model_dir, 'train'))
test_summary_writer = tf.summary.create_file_writer(os.path.join(model_dir, 'test'))

shutil.copy('config.json', model_dir)

# Initialize training time counter
training_time = 0.0

# Main training loop

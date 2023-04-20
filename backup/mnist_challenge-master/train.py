import tensorflow as tf
from datetime import datetime
import json
import os
import shutil
import numpy as np
from timeit import default_timer as timer

from model import Model
from pgd_attack import LinfPGDAttack

@tf.function
def evaluate(model, x, y):
    predictions = model(x)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, axis=1), tf.argmax(y, axis=1)), tf.float32))
    return accuracy

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
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
mnist_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
model = Model()

# Setting up the optimizer
optimizer = tf.optimizers.Adam(1e-4)

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
checkpoint = tf.train.Checkpoint(model=model)
saver = tf.train.CheckpointManager(checkpoint,'savers',max_to_keep=3)
train_accuracy_adv = tf.keras.metrics.Accuracy()
test_accuracy_adv = tf.keras.metrics.Accuracy()
xent_adv_train = tf.keras.metrics.CategoricalCrossentropy()
xent_adv = tf.keras.metrics.CategoricalCrossentropy()
'''tf.summary.scalar('accuracy adv train', train_accuracy_adv.result())
tf.summary.scalar('accuracy adv', test_accuracy_adv.result())
tf.summary.scalar('xent adv train', xent_adv_train.result())
tf.summary.scalar('xent adv', xent_adv.result())
tf.summary.image('images adv train', model.x_image)
merged_summaries = tf.compat.v1.summary.merge_all()'''


@tf.function
def train_step(x_batch, y_batch):
    # Initialize training time counter
    training_time = 0.0

        # Compute Adversarial Perturbations
    start = timer()
    x_batch_adv = attack.perturb(x_batch, y_batch)
    end = timer()
    training_time += end - start

    nat_dict = {model.x_input: x_batch,
                model.y_input: y_batch}

    adv_dict = {model.x_input: x_batch_adv,
                model.y_input: y_batch}

    with tf.GradientTape() as tape:
        # Compute logits for natural inputs
        logits_nat = model(nat_dict['x_input'], training=True)
        # Compute loss for natural inputs
        loss_nat = tf.keras.losses.categorical_crossentropy(
            nat_dict['y_input'], logits_nat, from_logits=True)
        loss_nat = tf.reduce_mean(loss_nat)

        # Compute logits for adversarial inputs
        logits_adv = model(adv_dict['x_input'], training=True)
        # Compute loss for adversarial inputs
        loss_adv = tf.keras.losses.categorical_crossentropy(
            adv_dict['y_input'], logits_adv, from_logits=True)
        loss_adv = tf.reduce_mean(loss_adv)

        # Compute total loss
        loss = loss_nat + loss_adv

    # Compute gradients and update weights
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Output to stdout
    if global_step % num_output_steps == 0:
        nat_acc = evaluate(model, nat_dict['.x_input'], nat_dict['y_input'])
        adv_acc = evaluate(model, adv_dict['x_input'], adv_dict['y_input'])
        print('Step {}:    ({})'.format(global_step, datetime.now()))
        print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
        print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
        if global_step != 0:
            print('    {} examples per second'.format(
                num_output_steps * batch_size / training_time))
            training_time = 0.0
        # Tensorboard summaries
        # da aggiungere
        if global_step % num_checkpoint_steps == 0:
            model.save(
                os.path.join(model_dir, 'checkpoint'),
                save_format='tf',
                options=tf.saved_model.SaveOptions(experimental_io_device='/job:localhost'))
    
    train_accuracy_adv.update_state(tf.argmax(nat_dict['y_input'], axis=1), tf.argmax(logits_nat, axis=1))
    test_accuracy_adv.update_state(tf.argmax(adv_dict['y_input'], axis=1), tf.argmax(logits_adv, axis=1))
    xent_adv_train.update_state(loss_nat/batch_size)
    xent_adv.update_state(loss_adv/batch_size)

    # Update global step
    global_step.assign_add(1)

training_time = 0
start = timer()
for x_train, y_train in mnist_train:
    train_step(x_train,y_train)

end = timer()

training_time += end - start


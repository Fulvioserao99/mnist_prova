import tensorflow as tf
from datetime import datetime
import json
import os
from model import Model
import numpy as np
from timeit import default_timer as timer
from pgd_attack import LinfPGDAttack

'''# Costruzione del modello
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (5,5), activation='relu', padding='same', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (5,5), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(10)
])'''
x_input = tf.keras.layers.Input(shape=(28,28,1), dtype=tf.float32)
y_input = tf.keras.layers.Input(shape=(10,), dtype=tf.int64)
x_image = tf.reshape(x_input, [-1, 28, 28, 1])
# convolutional layers
conv1 = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')(x_image)
pool1 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(conv1)
conv2 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same')(pool1)
pool2 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(conv2)
# flatten layer
flatten = tf.keras.layers.Flatten()(pool2)
# fully connected layers
fc1 = tf.keras.layers.Dense(1024, activation='relu')(flatten)
# output layer
output = tf.keras.layers.Dense(10)(fc1)

model = tf.keras.models.Model(inputs=[x_input], outputs=output)

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
x_train = np.expand_dims(x_train.astype(np.float32) / 255.0, axis=-1)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
x_test = np.expand_dims(x_test.astype(np.float32) / 255.0, axis=-1)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
mnist_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
mnist_train = mnist_train.shuffle(buffer_size=1024).batch(batch_size)
global_step = tf.Variable(0, trainable=False, dtype=tf.int64)



# Compilazione del modello
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



# Set up adversary
attack = LinfPGDAttack(model,
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])
saver = tf.keras.Model.save_weights(model,'savers/my_model_weights')


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
    print('prima della perturb')
    x_batch_adv = attack.perturb(x_batch,y_batch)
    end = timer()
    training_time += end - start

    x_key = x_input.ref()
    y_key = y_input.ref()
    nat_dict = {x_key: x_batch,
                y_key: y_batch}

    adv_dict = {x_key: x_batch_adv,
                y_key: y_batch}


    with tf.GradientTape() as tape:
        # Compute logits for natural inputs
        logits_nat = model(nat_dict['x_key'], training=True)
        # Compute loss for natural inputs
        loss_nat = tf.keras.losses.categorical_crossentropy(
            nat_dict['y_key'], logits_nat, from_logits=True)
        loss_nat = tf.reduce_mean(loss_nat)

        # Compute logits for adversarial inputs
        logits_adv = model(adv_dict['x_key'], training=True)
        # Compute loss for adversarial inputs
        loss_adv = tf.keras.losses.categorical_crossentropy(
            adv_dict['y_key'], logits_adv, from_logits=True)
        loss_adv = tf.reduce_mean(loss_adv)

        # Compute total loss
        loss = loss_nat + loss_adv

    # Compute gradients and update weights
    gradients = tape.gradient(loss, model.trainable_variables)

    # Output to stdout
    if global_step % num_output_steps == 0:
        nat_acc = evaluate(model, nat_dict['.x_key'], nat_dict['y_key'])
        adv_acc = evaluate(model, adv_dict['x_key'], adv_dict['y_key'])
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
    
    train_accuracy_adv.update_state(tf.argmax(nat_dict['y_key'], axis=1), tf.argmax(logits_nat, axis=1))
    test_accuracy_adv.update_state(tf.argmax(adv_dict['y_key'], axis=1), tf.argmax(logits_adv, axis=1))
    xent_adv_train.update_state(loss_nat/batch_size)
    xent_adv.update_state(loss_adv/batch_size)

    # Update global step
    global_step.assign_add(1)

training_time = 0
start = timer()
for step, (x_train, y_train) in enumerate(mnist_train):
    train_step(x_batch = x_train,y_batch = y_train)

end = timer()

training_time += end - start


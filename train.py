import tensorflow as tf
from datetime import datetime
import json
import os
 
from timeit import default_timer as timer
from pgd_attack import LinfPGDAttack
from pippo import MasterImage
from pippo import MasterImage2
    
import tensorflow as tf
from datetime import datetime
import json
    
# Model building
x_input = tf.keras.layers.Input(shape=(100, 100, 3), dtype=tf.float32)
y_input = tf.keras.layers.Input(shape=(10,), dtype=tf.int64)
conv1 = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')(x_input)
pool1 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(conv1)
conv2 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same')(pool1)
pool2 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(conv2)
flatten = tf.keras.layers.Flatten()(pool2)
fc1 = tf.keras.layers.Dense(1024, activation='relu')(flatten)
output = tf.keras.layers.Dense(10)(fc1)
model = tf.keras.models.Model(inputs=[x_input], outputs=output,name="my_model")

with open('config.json') as config_file:
    config = json.load(config_file)

# Setting up training parameters
tf.random.set_seed(config['random_seed'])
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
batch_size = config['training_batch_size']

# Setting up the data and the model
a = MasterImage(PATH=r'/home/lucapezz/Scrivania/Tirocinio/CHIDataset/test', IMAGE_SIZE=100)
b = MasterImage2(PATH=r'/home/lucapezz/Scrivania/Tirocinio/CHIDataset/training', IMAGE_SIZE=100)
(x_train, y_train) = a.load_dataset() 
(x_test, y_test) = b.load_dataset() 
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
#x_train = np.expand_dims(x_train.astype(np.float32) / 255.0, axis=-1)
y_train = tf.one_hot(y_train, depth=10)

x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
#x_test = np.expand_dims(x_test.astype(np.float32) / 255.0, axis=-1)
y_test = tf.one_hot(y_test, depth=10)

unisa_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
unisa_train = unisa_train.shuffle(buffer_size=1024).batch(batch_size)
global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# Compilazione del modello
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=tf.keras.metrics.Accuracy())

# Set up adversary
attack = LinfPGDAttack(model,
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])
saver = tf.keras.Model.save_weights(model, 'savers/my_model_weights')

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Setting up the metrics and the savers
checkpoint = tf.train.Checkpoint(model=model)
saver = tf.train.CheckpointManager(checkpoint, 'savers', max_to_keep=3)
test_accuracy_adv = tf.keras.metrics.Accuracy() 
test_accuracy_nat = tf.keras.metrics.Accuracy()
writer = tf.summary.create_file_writer("tmp/mylogs")

@tf.function
def training(x_batch, y_batch):
    with tf.GradientTape() as tape:
        # Compute logits 
        logits = model(x_batch, training=True)
        # Compute loss 
        loss = model.loss(y_batch, logits)
        
    # Compute gradients and update weights
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return logits, loss


@tf.function
def acc_calc(logits_nat, logits_adv, y_batch):
    test_accuracy_adv.update_state(tf.argmax(y_batch, axis=1), tf.argmax(logits_adv, axis=1))
    test_accuracy_nat.update_state(tf.argmax(y_batch, axis=1), tf.argmax(logits_nat, axis=1))


def train_step(x_batch, x_batch_adv, y_batch, time, step):
    logits_adv, loss_adv = training(x_batch_adv, y_batch)
    logits_nat, loss_nat = training(x_batch, y_batch)

    # Output to stdout
    if step % num_output_steps == 0:
        print('\n-- SIAMO AL CHECKPOINT --')
        acc_calc(logits_nat, logits_adv, y_batch)
        tf.print('Loss nat:', loss_nat)
        tf.print('Loss adv:', loss_adv)
        print('Step {}:    ({})'.format(step, datetime.now()))
        tf.print("Training nat_acc over epoch: ", float(test_accuracy_nat.result() * 100))
        tf.print("Training adv_acc over epoch: ", float(test_accuracy_adv.result() * 100))
        
        if step != 0:
            print('{} examples per second'.format(num_output_steps * batch_size / time))
        
        time = 0.0
    
    # Tensorboard summaries
    if step % num_summary_steps == 0:
        with writer.as_default():
            tf.summary.scalar("loss_nat", loss_nat, step=step)
            tf.summary.scalar("loss_adv", loss_adv, step=step)
            tf.summary.scalar("acc_nat", test_accuracy_nat.result(), step=step)
            tf.summary.scalar("acc_adv", test_accuracy_adv.result(), step=step)
        
    if step % num_checkpoint_steps == 0:
        tf.train.Checkpoint(model=model)

    
training_time = 0
start = timer()
for epoch in range(10):
    print("\n---------------------Start of epoch %d---------------------" % epoch) 
    for step, (x_train, y_train) in enumerate(unisa_train):
        t_time = 0
        start_t = timer()
        x_batch_adv = attack.perturb(x_train, y_train)
        end_t = timer()
        t_time += end_t - start_t
        train_step(x_batch=x_train, y_batch=y_train, x_batch_adv=x_batch_adv, time=t_time, step=step)
        test_accuracy_adv.reset_states()
        test_accuracy_nat.reset_states()

        with writer.as_default():
            tf.summary.image("adv_images", x_batch_adv, step=global_step, max_outputs=batch_size)
            tf.summary.image("nat_images", x_train, step=global_step, max_outputs=batch_size)

        writer.flush()

    end = timer()
    training_time += end - start


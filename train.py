import tensorflow as tf
from datetime import datetime
import json
import os
import numpy as np
from timeit import default_timer as timer
from dataset_input import MasterImage
from dataset_input import MasterImage2
from src import prnu
from multiprocessing import cpu_count, Pool
from PIL import Image
import os   
import random


    
# Model building
x_input = tf.keras.layers.Input(shape=(100, 100, 3), dtype=tf.float32)
y_input = tf.keras.layers.Input(shape=(20,), dtype=tf.int64)
conv1 = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')(x_input)
pool1 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(conv1)
conv2 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same')(pool1)
pool2 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(conv2)
flatten = tf.keras.layers.Flatten()(pool2)
fc1 = tf.keras.layers.Dense(1024, activation='relu')(flatten)
output = tf.keras.layers.Dense(20)(fc1)
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
train_ds = MasterImage(PATH=r'E:\dataset\training', IMAGE_SIZE=100)
test_ds = MasterImage2(PATH=r'E:\dataset\testing', IMAGE_SIZE=100)
(x_train, y_train) = train_ds.load_dataset() 
(x_test, y_test) = test_ds.load_dataset() 

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.one_hot(y_train, depth=20)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_test = tf.one_hot(y_test, depth=20)

unisa_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
unisa_train = unisa_train.shuffle(buffer_size=1024).batch(batch_size)
global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# Compilazione del modello
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=tf.keras.metrics.Accuracy())


# Setting up the metrics and the savers
checkpoint = tf.train.Checkpoint(model=model)
saver = tf.train.CheckpointManager(checkpoint, 'savers', max_to_keep=3)
test_accuracy_adv = tf.keras.metrics.Accuracy()
test_accuracy_nat = tf.keras.metrics.Accuracy()
writer = tf.summary.create_file_writer("tmp/mylogs")
latest_checkpoint = saver.latest_checkpoint

if latest_checkpoint:
    checkpoint.restore(latest_checkpoint)
    print("Modello ripristinato da:", latest_checkpoint)
    
else:
    print("Nessun checkpoint trovato.")



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

    with writer.as_default():
        tf.summary.image("adv_images", x_batch_adv, step=global_step, max_outputs=batch_size)
        tf.summary.image("nat_images", x_train, step=global_step, max_outputs=batch_size)

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
        saver.save()


def process_batch_of_images(batch, levels=4, sigma=5):
    """
    Process a batch of images using extract_single_aligned function.
    :param batch: Input batch of images of shape (B, H, W, Ch) and type np.uint8.
    :param levels: Number of wavelet decomposition levels.
    :param sigma: Estimated noise power.
    :return: TensorFlow tensor of PRNU for each image in the batch.
    """
    assert isinstance(batch, np.ndarray)
    assert batch.ndim == 4

    prnu_list = []
    batch_uint8 = batch.astype(np.uint8)

    for img in batch_uint8:
        prnu = extract_aligned_s(img, levels, sigma)
        prnu_list.append(prnu)

    prnu_tensor = tf.convert_to_tensor(prnu_list, dtype=tf.float32)

    return prnu_tensor

def extract_aligned_s(img, levels=2, sigma=5):
    """
    Extract PRNU from a single image.
    :param img: Input image of shape (H, W, Ch) and type np.uint8.
    :param levels: Number of wavelet decomposition levels.
    :param sigma: Estimated noise power.
    :return: PRNU
    """
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3
    
    img = img.astype(np.uint8)

    h, w, ch = img.shape

    RPsum = np.zeros((h, w, ch), np.float32)
    NN = np.zeros((h, w, ch), np.float32)

    nni = prnu.inten_sat_compact((img, levels, sigma))
    NN += nni

    wi = prnu.noise_extract_compact((img, levels, sigma))
    RPsum += wi

    K = RPsum / (NN + 1)
    K = prnu.rgb2gray(K)
    K = prnu.zero_mean_total(K)
    K = prnu.wiener_dft(K, K.std(ddof=1)).astype(np.float32)

    return K


# Directory contenente le cartelle con le immagini
base_directory = r'E:\dataset\training'

# Inizializza un array vuoto per memorizzare le prime immagini di ogni cartella
prnu_sample = []

# Scansione delle sottodirectory
for cartella in os.listdir(base_directory):
    cartella_completa = os.path.join(base_directory, cartella)
    
    # Verifica se il percorso è una cartella
    if os.path.isdir(cartella_completa):
        # Trova tutti i file nella cartella
        file_in_cartella = os.listdir(cartella_completa)
        
        # Filtra solo i file con estensione immagine (ad esempio, .jpg o .png)
        immagini = [f for f in file_in_cartella if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Se ci sono immagini nella cartella, aggiungi la prima all'array
        if immagini:
            prima_immagine = Image.open(os.path.join(cartella_completa, immagini[0]))
            prima_immagine = tf.image.resize(prima_immagine, (100, 100))
            prima_immagine = np.array(prima_immagine)
            prnu_img = extract_aligned_s(prima_immagine)
            prnu_sample.append(prnu_img)

prnu_sample = np.array([np.array(img) for img in prnu_sample])

img = tf.random.uniform((100, 100, 3))

training_time = 0
start = timer()
for epoch in range(10):
    print("\n---------------------Start of epoch %d---------------------" % epoch) 
    for step,(x_train, y_train) in enumerate(unisa_train):
        t_time = 0
        start_t = timer()
        rand_index = random.randint(0, len(prnu_sample) - 1)
        rand_img = prnu_sample[rand_index]
        img = rand_img
        # Espandi img_da_sommare per avere una dimensione batch e canali
        img_exp = tf.expand_dims(img, axis=0)  # Aggiunge una dimensione batch virtuale
        img_exp = tf.expand_dims(img_exp, axis=-1)  # Aggiunge una dimensione canali virtuale
        batch_sz = x_train.shape[0]
        img_exp = tf.tile(img_exp, [batch_sz, 1, 1, 3])
        x_batch_adv = x_train + img_exp
        end_t = timer()
        t_time += end_t - start_t
        train_step(x_batch=x_train, y_batch=y_train, x_batch_adv=x_batch_adv, time=t_time, step=step)
    test_accuracy_adv.reset_states()
    test_accuracy_nat.reset_states()
    writer.flush()

end = timer()
training_time += end - start


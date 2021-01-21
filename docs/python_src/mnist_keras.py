import tensorflow as tf
import tensorflow_datasets as tfds

n_rounds = 20
width = 28
height = 28
n_classes = 10
l_rate = 0.001
batch_size = 64

# Load the data
train_dataset = tfds.load('mnist', split='train', as_supervised=True)
test_dataset = tfds.load('mnist', split='test', as_supervised=True)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


train_dataset = train_dataset.map(normalize_img,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(len(train_dataset))
train_dataset = train_dataset.batch(batch_size)

test_dataset = test_dataset.map(normalize_img,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.shuffle(len(test_dataset))
test_dataset = test_dataset.batch(batch_size)

# Define the model
input_img = tf.keras.Input(shape=(width, height, 1), name="Input")
x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="Conv1_1")(input_img)
x = tf.keras.layers.BatchNormalization(name="bn1")(x)
x = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="Conv2_1")(x)
x = tf.keras.layers.BatchNormalization(name="bn4")(x)
x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")(x)
x = tf.keras.layers.Flatten(name="flatten")(x)
x = tf.keras.layers.Dense(n_classes, activation="softmax", name="fc1")(x)
model = tf.keras.Model(inputs=input_img, outputs=x)

opt = tf.keras.optimizers.Adam(lr=l_rate)
model.compile(
    loss="sparse_categorical_crossentropy",
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    optimizer=opt)

# Train and evaluate model
for round in range(n_rounds):
    model.fit(train_dataset, steps_per_epoch=40)
    result = model.evaluate(x=test_dataset, return_dict=True, steps=10)
    print(f"Performance at round {round} is {result}")

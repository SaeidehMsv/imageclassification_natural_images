import tensorflow as tf
from model import create_model
import numpy as np
from PIL import Image

path = '\\natural_images'
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=path,
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
)
# type(train_ds)
# print(type(train_ds))
# print(train_ds.labels)
for element in train_ds:
    print(element[1][0])
    im = Image.fromarray(np.uint8(element[0][0]))
    im.show()
    break
model = create_model()
model.summary()
history = model.fit(train_ds, epochs=50, batch_size=64, verbose=1, shuffle=1)

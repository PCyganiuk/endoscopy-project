import pandas as pd
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models

labels_path = "/mnt/e/ERS/ers/labels.csv"
label_names = "/mnt/e/ERS/ers/names.json"
base_dir = "/mnt/e/ERS/ers/"

df = pd.read_csv(labels_path, sep="\t")
df.head()

multi_labels = (
    df.dropna(subset=["label"])
      .groupby("file")["label"]
      .apply(list)
      .reset_index()
)

multi_labels.head()

with open(label_names) as f:
    code_to_name = json.load(f)

code_to_idx = {code: i for i, code in enumerate(sorted(code_to_name.keys()))}
num_classes = len(code_to_idx)

def make_multihot(label_list, num_classes=num_classes):
    vec = np.zeros(num_classes, dtype=np.float32)
    for code in label_list:
        if code in code_to_idx:
            vec[code_to_idx[code]] = 1.0
    return vec

multi_labels["multi_hot"] = multi_labels["label"].apply(lambda x: make_multihot(x, num_classes))

multi_labels = (
    df.dropna(subset=["label"])
      .groupby("file")["label"]
      .apply(list)
      .reset_index()
)

multi_labels["multi_hot"] = multi_labels["label"].apply(
    lambda x: make_multihot(x, num_classes)
)

image_paths = [base_dir + f for f in multi_labels["file"].tolist()]
multi_hot_labels = np.stack(multi_labels["multi_hot"].values)

def preprocess_with_padding(image_path, label=None):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.resize_with_pad(img, 224, 224)

    img = tf.cast(img, tf.float32) / 255.0

    if label is None:
        return img
    return img, label

dataset_labeled = tf.data.Dataset.from_tensor_slices((image_paths, multi_hot_labels))
dataset_labeled = dataset_labeled.map(preprocess_with_padding).batch(8).shuffle(100)
base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
    pooling="avg"
)

model = models.Sequential([
    base_model,
    layers.Dense(num_classes, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"]
)

history = model.fit(dataset_labeled,epochs=10)
model.save("/mnt/e/ERS/ers/resnet50_multilabel.h5")

#preds = model.predict(dataset_unlabeled)
#pseudo_labels = (preds > 0.9).astype(int)
#X_combined = np.concatenate([dataset_labeled, X_unlabeled_selected])
#Y_combined = np.concatenate([Y_labeled, pseudo_labels_selected])
#model.fit(X_combined, Y_combined)
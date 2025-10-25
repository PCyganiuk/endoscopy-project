import tensorflow as tf
import numpy as np

from tensorflow.keras import layers, models
from src.classes import Classes
from src.dataset_loader import DatasetLoader

class Models:

    def __init__(self, args):
        self.args = args
        self.mode = args.type_num

    def train(self):

        if self.mode == 0:
            self.baseline()

    def baseline(self):
        num_classes = Classes().num_classes
        dataset_loader = DatasetLoader(self.args)
        self.ers_labeled_all, self.ers_labels_all, self.ers_unlabeled_all = dataset_loader.prepare_ers()
        self.ers_labeled_train, self.ers_labels_train, self.ers_labeled_test, self.ers_labels_test, self.ers_unlabeled = dataset_loader.split_labeled_by_patient_id(self.ers_labeled_all, self.ers_labels_all, self.ers_unlabeled_all, self.args.test_size)
        dataset_labeled_train = tf.data.Dataset.from_tensor_slices((self.ers_labeled_train, self.ers_labels_train))
        dataset_labeled_train = dataset_labeled_train.map(self.preprocess_with_padding).batch(8).shuffle(100)

        dataset_labeled_test = tf.data.Dataset.from_tensor_slices((self.ers_labeled_test, self.ers_labels_test))
        dataset_labeled_test = dataset_labeled_test.map(self.preprocess_with_padding).batch(8)
        
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

        preds = model.fit(dataset_labeled_train,epochs=1)

        print("Generating pseudo-labels for ERS unlabeled data")
        dataset_unlabeled = tf.data.Dataset.from_tensor_slices(self.ers_unlabeled)
        dataset_unlabeled = dataset_unlabeled.map(self.preprocess_with_padding).batch(8)

        preds = model.predict(dataset_unlabeled, verbose=1)
        pseudo_labels = (preds > 0.9).astype(np.float32)

        mask_confident = np.max(preds, axis=1) > 0.9
        ers_confident_images = np.array(self.ers_unlabeled)[mask_confident]
        ers_confident_labels = pseudo_labels[mask_confident]

        print(f"Selected {len(ers_confident_images)} pseudo-labeled ERS samples")

        X_combined = np.concatenate([self.ers_labeled_train, ers_confident_images])
        Y_combined = np.concatenate([self.ers_labels_train, ers_confident_labels])

        dataset_combined = tf.data.Dataset.from_tensor_slices((X_combined, Y_combined))
        dataset_combined = dataset_combined.map(self.preprocess_with_padding).batch(8).shuffle(200)

        print("Retraining on labeled + pseudo-labeled ERS data")
        model.fit(dataset_combined, epochs=1)

        model.save("/mnt/e/ERS/ers/resnet50_semisupervised_multilabel.h5")
        print("Semi-supervised ERS training completed.")
        
        results = model.evaluate(dataset_labeled_test, verbose=1)
        for name, value in zip(model.metrics_names, results):
            print(f"  {name}: {value:.4f}")
        


    
    def preprocess_with_padding(self, image_path, label=None):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)

        img = tf.image.resize_with_pad(img, 224, 224)

        img = tf.cast(img, tf.float32) / 255.0

        if label is None:
            return img
        return img, label
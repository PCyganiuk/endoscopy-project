import tensorflow as tf
import numpy as np
from datetime import datetime
import os

from tensorflow.keras import layers, models # type: ignore
from src.classes import Classes
from src.dataset_loader import DatasetLoader
from tensorflow.keras.callbacks import CSVLogger # type: ignore

class Models:

    def __init__(self, args):
        self.args = args
        self.mode = args.type_num
        self.k = args.k_folds
        self.epochs = args.epochs
        self.model_size = args.model_size
        self.verbose = args.verbose

    def train(self):

        if self.mode == 0:
            self.baseline()
        elif self.mode == 1:
            self.train_ers_test_galar() # train on ers test on galar
        elif self.mode == 2:
            self.train_galar_test_ers() # train on galar test on ers

    def get_backbone(self):
        if self.model_size == 0:
            base_model = tf.keras.applications.MobileNetV2(
                include_top=False,
                weights="imagenet",
                input_shape=(224, 224, 3),
                pooling="avg"
            )
        elif self.model_size == 1:
            base_model = tf.keras.applications.DenseNet121(
                include_top=False,
                weights="imagenet",
                input_shape=(224, 224, 3),
                pooling="avg"
            )
        elif self.model_size == 2:
            base_model = tf.keras.applications.ResNet50(
                include_top=False,
                weights="imagenet",
                input_shape=(224, 224, 3),
                pooling="avg"
            )
        return base_model

    def baseline(self):
        num_classes = Classes().num_classes
        dataset_loader = DatasetLoader(self.args)
        self.ers_labeled_all, self.ers_labels_all, self.ers_unlabeled_all = dataset_loader.prepare_ers()

        kf_generator = dataset_loader.split_patients_kfold(
            self.ers_labeled_all, 
            self.ers_labels_all, 
            self.ers_unlabeled_all, 
            self.k
            )

        for fold, (ers_labeled_train, ers_labels_train, ers_labeled_test, ers_labels_test, _) in enumerate(kf_generator, 1):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            num_classes = Classes().num_classes
            os.makedirs("logs", exist_ok=True)
            csv_logger = CSVLogger(f"logs/csv/baseline_training_log_model_{self.model_size}_fold_{fold}_{timestamp}.csv", append=True)
            
            print(f"\n================ Fold {fold}/{self.k} ================")
            print(f"Train samples: {len(ers_labeled_train)} | Test samples: {len(ers_labeled_test)}")

            model = self.build_model(num_classes)

            model.fit(
                self.make_dataset(ers_labeled_train, ers_labels_train, shuffle=True),
                validation_data=self.make_dataset(ers_labeled_test, ers_labels_test), 
                epochs=self.epochs,
		        verbose=self.verbose, 
                callbacks=[csv_logger]
                )


            save_path = f"weights/baseline_{self.model_size}_fold{fold}.h5"
            model.save(save_path)
            print(f"Saved {self.model_size} model for fold {fold} -> {save_path}")


    def train_ers_test_galar(self):
        num_classes = Classes().num_classes
        dataset_loader = DatasetLoader(self.args)
        ers_labeled_all, ers_labels_all, ers_unlabeled_all = dataset_loader.prepare_ers()
        galar_images, galar_labels = dataset_loader.prepare_galar()

        kf_gen = dataset_loader.split_patients_kfold(ers_labeled_all, ers_labels_all, ers_unlabeled_all, self.k)

        for fold, (ers_labeled_train, ers_labels_train, _, _, _) in enumerate(kf_gen, 1):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            csv_logger = CSVLogger(f"logs/csv/ers2galar_{self.model_size}_fold{fold}_{timestamp}.csv", append=True)

            print(f"\n================ Fold {fold}/{self.k} ================")
            print(f"Train samples: {len(ers_labeled_train)} | Test samples: {len(galar_images)}")

            model = self.build_model(num_classes)
            model.fit(
                self.make_dataset(ers_labeled_train, ers_labels_train, shuffle=True),
                validation_data=self.make_dataset(galar_images, galar_labels),
                epochs=self.epochs,
		        verbose=self.verbose, 
                callbacks=[csv_logger]
            )
            os.makedirs("weights", exist_ok=True)
            model.save(f"weights/ers2galar_{self.model_size}_fold{fold}_{timestamp}.h5")


    def train_galar_test_ers(self):
        num_classes = Classes().num_classes
        dataset_loader = DatasetLoader(self.args)
        galar_images, galar_labels = dataset_loader.prepare_galar()
        ers_labeled_all, ers_labels_all, _ = dataset_loader.prepare_ers()

        kf_gen = dataset_loader.split_patients_kfold(galar_images, galar_labels, None, self.k)

        for fold, (galar_train, galar_labels_train, galar_test, galar_labels_test, _) in enumerate(kf_gen, 1):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            csv_logger = CSVLogger(f"logs/csv/galar2ers_{self.model_size}_fold{fold}_{timestamp}.csv", append=True)

            model = self.build_model(num_classes)
            model.fit(
                self.make_dataset(galar_train, galar_labels_train, shuffle=True),
                validation_data=self.make_dataset(ers_labeled_all, ers_labels_all),
                epochs=self.epochs,
		        verbose=self.verbose, 
                callbacks=[csv_logger]
            )
            os.makedirs("weights", exist_ok=True)
            model.save(f"weights/galar2ers_{self.model_size}_fold{fold}_{timestamp}.h5")
    
    def build_model(self, num_classes):
        base_model = self.get_backbone()
        model = models.Sequential([
            base_model,
            layers.Dense(num_classes, activation="sigmoid")
        ])
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc")
            ]
        )
        return model
    
    def make_dataset(self, images, labels=None, shuffle=False, batch_size=8):
        ds = tf.data.Dataset.from_tensor_slices((images, labels) if labels is not None else images)
        if labels is not None:
            ds = ds.map(self.preprocess_with_padding)
        else:
            ds = ds.map(lambda x: self.preprocess_with_padding(x))
        if shuffle:
            ds = ds.shuffle(100)
        return ds.batch(batch_size)

    
    def preprocess_with_padding(self, image_path, label=None):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)

        img = tf.image.resize_with_pad(img, 224, 224)

        img = tf.cast(img, tf.float32) / 255.0

        if label is None:
            return img
        return img, label
    

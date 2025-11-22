import tensorflow as tf
import numpy as np
from datetime import datetime
import os

from tensorflow.keras import layers, models # type: ignore
from src.classes import Classes
from src.f1_score import F1Score
from src.optimal_treshhold_callback import OptimalThresholdCallback
from src.dataset_loader import DatasetLoader
from tensorflow.keras.callbacks import CSVLogger # type: ignore

class Models:

    def __init__(self, args):
        self.args = args
        self.mode = args.type_num
        self.k = args.k_folds
        self.epochs = args.epochs
        self.model_size = args.model_size
        self.binary = args.binary
        self.verbose = args.verbose

    def train(self):

        if self.mode == 0:
            self.baseline() # train on ers test on ers
        elif self.mode == 1:
            self.train_ers_test_galar() # train on ers test on galar
        elif self.mode == 2:
            self.train_galar_test_ers() # train on galar test on ers
        elif self.mode == 3:
            self.train_galar_test_galar() # train on galar test on galar
        elif self.mode == 4:
            self.train_ersXgalar_test_ersORgalar() # train on ers x galar test on ers and galar separetely

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
        num_classes = Classes(self.binary).num_classes
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
            os.makedirs("logs", exist_ok=True)
            csv_logger = CSVLogger(f"logs/csv/baseline_training_log_model_{self.model_size}_fold_{fold}_{timestamp}.csv", append=True)
            
            print(f"\n================ Fold {fold}/{self.k} ================")
            print(f"Train samples: {len(ers_labeled_train)} | Test samples: {len(ers_labeled_test)}")

            model = self.build_model(num_classes)
            val_ds = self.make_dataset(ers_labeled_test, ers_labels_test, val=True)
            model.fit(
                self.make_dataset(ers_labeled_train, ers_labels_train, shuffle=True, val=False),
                validation_data=val_ds, 
                epochs=self.epochs,
		        verbose=self.verbose, 
                callbacks=[
                    OptimalThresholdCallback(val_ds, name="val"),
                    csv_logger
                ]
            )


            save_path = f"weights/baseline_{self.model_size}_fold{fold}.h5"
            model.save(save_path)
            print(f"Saved {self.model_size} model for fold {fold} -> {save_path}")


    def train_ers_test_galar(self):
        num_classes = Classes(self.binary).num_classes
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
            val_ds = self.make_dataset(galar_images, galar_labels, val=True)
            model.fit(
                self.make_dataset(ers_labeled_train, ers_labels_train, shuffle=True, val=False),
                validation_data=val_ds,
                epochs=self.epochs,
		        verbose=self.verbose, 
                callbacks=[
                           OptimalThresholdCallback(val_ds),
                           csv_logger,
                ]
            )
            os.makedirs("weights", exist_ok=True)
            model.save(f"weights/ers2galar_{self.model_size}_fold{fold}_{timestamp}.h5")


    def train_galar_test_ers(self):
        num_classes = Classes(self.binary).num_classes
        dataset_loader = DatasetLoader(self.args)
        galar_images, galar_labels = dataset_loader.prepare_galar()
        ers_labeled_all, ers_labels_all, _ = dataset_loader.prepare_ers()

        kf_gen = dataset_loader.split_patients_kfold(galar_images, galar_labels, None, self.k)

        for fold, (galar_train, galar_labels_train, galar_test, galar_labels_test, _) in enumerate(kf_gen, 1):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            csv_logger = CSVLogger(f"logs/csv/galar2ers_{self.model_size}_fold{fold}_{timestamp}.csv", append=True)

            model = self.build_model(num_classes)
            val_ds = self.make_dataset(ers_labeled_all, ers_labels_all, val=True)
            model.fit(
                self.make_dataset(galar_train, galar_labels_train, shuffle=True, val=False),
                validation_data=val_ds,
                epochs=self.epochs,
		        verbose=self.verbose, 
                callbacks=[
                           OptimalThresholdCallback(val_ds),
                           csv_logger
                           ]
            )
            os.makedirs("weights", exist_ok=True)
            model.save(f"weights/galar2ers_{self.model_size}_fold{fold}_{timestamp}.h5")

    def train_galar_test_galar(self):
        num_classes = Classes(self.binary).num_classes
        dataset_loader = DatasetLoader(self.args)
        galar_images, galar_labels = dataset_loader.prepare_galar()

        kf_gen = dataset_loader.split_patients_kfold(galar_images, galar_labels, None, self.k)

        for fold, (galar_train, galar_labels_train, galar_test, galar_labels_test, _) in enumerate(kf_gen, 1):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            csv_logger = CSVLogger(f"logs/csv/galar2galar_{self.model_size}_fold{fold}_{timestamp}.csv", append=True)

            model = self.build_model(num_classes)
            val_ds = self.make_dataset(galar_test, galar_labels_test, val=True)
            model.fit(
                self.make_dataset(galar_train, galar_labels_train, shuffle=True, val=False),
                validation_data=val_ds,
                epochs=self.epochs,
		        verbose=self.verbose, 
                callbacks=[
                           OptimalThresholdCallback(val_ds),
                           csv_logger
                          ]
            )
            os.makedirs("weights", exist_ok=True)
            model.save(f"weights/galar2galar_{self.model_size}_fold{fold}_{timestamp}.h5")
        
    def train_ersXgalar_test_ersORgalar(self):
        num_classes = Classes(self.binary).num_classes
        dataset_loader = DatasetLoader(self.args)

        galar_images, galar_labels = dataset_loader.prepare_galar()
        ers_images, ers_labels, _ = dataset_loader.prepare_ers()

        kf_gen_ers = dataset_loader.split_patients_kfold(ers_images, ers_labels, None, self.k)
        kf_gen_galar = dataset_loader.split_patients_kfold(galar_images, galar_labels, None, self.k)

        for fold, ((ers_train, ers_labels_train, ers_val, ers_labels_val, _),
                (galar_train, galar_labels_train, galar_val, galar_labels_val, _)) in enumerate(zip(kf_gen_ers, kf_gen_galar), 1):

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            os.makedirs("logs/csv", exist_ok=True)
            csv_logger = CSVLogger(
                f"logs/csv/ersXgalar_test_ersORgalar_{self.model_size}_fold{fold}_{timestamp}.csv",
                append=True
            )

            train_images = np.concatenate([ers_train, galar_train])
            train_labels = np.concatenate([ers_labels_train, galar_labels_train])

            val_ds_ers = self.make_dataset(ers_val, ers_labels_val, val=True)
            val_ds_galar = self.make_dataset(galar_val, galar_labels_val, val=True)

            model = self.build_model(num_classes)

            model.fit(
                self.make_dataset(train_images, train_labels, shuffle=True, val=False),
                validation_data=val_ds_ers,
                epochs=self.epochs,
                verbose=self.verbose,
                callbacks=[
                    OptimalThresholdCallback(val_ds_ers, name="ERS"),
                    OptimalThresholdCallback(val_ds_galar, name="GALAR"),
                    csv_logger
                ]
            )

            os.makedirs("weights", exist_ok=True)
            model.save(f"weights/ersXgalar_test_ersORgalar_{self.model_size}_fold{fold}_{timestamp}.h5")


    
    def build_model(self, num_classes):
        base_model = self.get_backbone()
        if num_classes == 2:
            activation = "softmax"
            loss = tf.keras.losses.CategoricalCrossentropy()
        else:
            activation = "sigmoid"
            loss = "binary_crossentropy"

        model = models.Sequential([
            base_model,
            layers.Dense(num_classes, activation=activation)
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="sensitivity", class_id=1),
                tf.keras.metrics.Recall(name="specificity", class_id=0),
                tf.keras.metrics.AUC(name="auc"),
                F1Score(name="f1")
            ]
        )
        return model
    
    def make_dataset(self, images, labels=None, shuffle=False, batch_size=32, val=False):
        ds = tf.data.Dataset.from_tensor_slices((images, labels) if labels is not None else images)
        if labels is not None:
            if val:
                ds = ds.map(self.preprocess_val)
            else:
                ds = ds.map(self.preprocess_with_padding)
        else:
            if val:
                ds = ds.map(lambda x: self.preprocess_val(x))
            else:
                ds = ds.map(lambda x: self.preprocess_with_padding(x))
        #if shuffle:
        #    ds = ds.shuffle(100)
        return ds.batch(batch_size)

    
    def preprocess_with_padding(self, image_path, label=None):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)

        img = tf.image.resize_with_pad(img, 224, 224)
        
        img = tf.cast(img, tf.float32) / 255.0

        img = (img - 0.5) * 2.0

        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
        img = tf.image.random_hue(img, max_delta=0.05)

        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)

        if label is None:
            return img
        return img, label
    
    def preprocess_val(self, image_path, label=None):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)

        img = tf.image.resize_with_pad(img, 224, 224)
        
        img = tf.cast(img, tf.float32) / 255.0

        img = (img - 0.5) * 2.0

        if label is None:
            return img
        return img, label
    

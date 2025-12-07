import tensorflow as tf
import numpy as np
from datetime import datetime
import os

from tensorflow.keras import layers, models # type: ignore
from src.classes import Classes
from src.f1_score import F1Score
from src.validation_metrics_callback import ValidationMetricsCallback
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
        self.fisheye=args.fisheye

    def train(self):
        os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
        os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir="
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        if self.mode == 0:
            self.train_ers_test_ersORgalar()
        elif self.mode == 1:
            self.train_galar_test_ersORgalar()
        elif self.mode == 2:
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
                f"logs/csv/ersXgalar_test_ersORgalar_{self.model_size}_fold_{fold}_{timestamp}.csv",
                append=True
            )

            train_images_ers = ers_train
            train_labels_ers = ers_labels_train
            train_images_galar = galar_train
            train_labels_galar = galar_labels_train

            train_ds_ers = self.make_dataset(train_images_ers, train_labels_ers, shuffle=False, val=False, ers=self.fisheye, fold=fold)

            train_ds_galar = self.make_dataset(train_images_galar, train_labels_galar, shuffle=False, val=False, ers=False, fold=fold)

            train_ds = train_ds_ers.concatenate(train_ds_galar)

            val_ds_ers = self.make_dataset(ers_val, ers_labels_val, val=True, ers=self.fisheye, fold=fold)
            val_ds_galar = self.make_dataset(galar_val, galar_labels_val, val=True, ers=False, fold=fold)
            model = self.build_model(num_classes)

            model.fit(
                train_ds,
                validation_data=val_ds_ers,
                epochs=self.epochs,
                verbose=self.verbose,
                callbacks=[
                    ValidationMetricsCallback(val_ds_galar, name="GALAR_val"),
                    csv_logger
                ]
            )

            os.makedirs("weights", exist_ok=True)
            model.save(f"weights/ersXgalar_test_ersORgalar_{self.model_size}_fold{fold}_{timestamp}.h5")

    def train_ers_test_ersORgalar(self):
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
                f"logs/csv/ers_test_ersORgalar_{self.model_size}_fold_{fold}_{timestamp}.csv",
                append=True
            )

            train_images_ers = ers_train
            train_labels_ers = ers_labels_train
            train_images_galar = galar_train
            train_labels_galar = galar_labels_train

            train_ds_ers = self.make_dataset(train_images_ers, train_labels_ers, shuffle=False, val=False, ers=self.fisheye, fold=fold)

            #train_ds_galar = self.make_dataset(train_images_galar, train_labels_galar, shuffle=False, val=False, ers=False, fold=fold)
            train_ds = train_ds_ers
            #train_ds = train_ds.shuffle(buffer_size=1000)

            val_ds_ers = self.make_dataset(ers_val, ers_labels_val, val=True, ers=self.fisheye, fold=fold)
            val_ds_galar = self.make_dataset(galar_val, galar_labels_val, val=True, ers=False, fold=fold)
            
            model = self.build_model(num_classes)

            model.fit(
                train_ds,
                validation_data=val_ds_ers,
                epochs=self.epochs,
                verbose=self.verbose,
                callbacks=[
                    ValidationMetricsCallback(val_ds_galar, name="GALAR_val"),
                    csv_logger
                ]
            )

            os.makedirs("weights", exist_ok=True)
            model.save(f"weights/ers_test_ersORgalar_{self.model_size}_fold{fold}_{timestamp}.h5")

    def train_galar_test_ersORgalar(self):
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
                f"logs/csv/galar_test_ersORgalar_{self.model_size}_fold_{fold}_{timestamp}.csv",
                append=True
            )

            #train_images_ers = ers_train
            #train_labels_ers = ers_labels_train
            train_images_galar = galar_train
            train_labels_galar = galar_labels_train

            #train_ds_ers = self.make_dataset(train_images_ers, train_labels_ers, shuffle=False, val=False, ers=self.fisheye, fold=fold)

            train_ds_galar = self.make_dataset(train_images_galar, train_labels_galar, shuffle=False, val=False, ers=False, fold=fold)

            train_ds = train_ds_galar

            val_ds_ers = self.make_dataset(ers_val, ers_labels_val, val=True, ers=self.fisheye, fold=fold)
            val_ds_galar = self.make_dataset(galar_val, galar_labels_val, val=True, ers=False, fold=fold)
            model = self.build_model(num_classes)

            model.fit(
                train_ds,
                validation_data=val_ds_ers,
                epochs=self.epochs,
                verbose=self.verbose,
                callbacks=[
                    ValidationMetricsCallback(val_ds_galar, name="GALAR_val"),
                    csv_logger
                ]
            )

            os.makedirs("weights", exist_ok=True)
            model.save(f"weights/galar_test_ersORgalar_{self.model_size}_fold{fold}_{timestamp}.h5")
    
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
    
    def make_dataset(self, images, labels=None, shuffle=False, val=False, ers=False, fold=1):
        AUTOTUNE = tf.data.AUTOTUNE
        ds = tf.data.Dataset.from_tensor_slices((images, labels) if labels is not None else images)
        if not val:
            ds = ds.shuffle(buffer_size=8192, reshuffle_each_iteration=True)
        if labels is not None:
            if val:
                ds = ds.map(lambda x, y: self.preprocess_val(x, y, ers=ers),num_parallel_calls=4,)
            else:
                ds = ds.map(lambda x, y: self.preprocess_with_padding(x, y, ers=ers),num_parallel_calls=4,)
        else:
            if val:
                ds = ds.map(lambda x: self.preprocess_val(x, ers=ers),num_parallel_calls=4,)
            else:
                ds = ds.map(lambda x: self.preprocess_with_padding(x, ers=ers),num_parallel_calls=4,)
        ds = ds.batch(512)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    
    def preprocess_with_padding(self, image_path, label=None, ers=False):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        img = tf.image.resize_with_pad(img, 224, 224)

        if ers:
            img = self.fisheye_tf(img, zoom_factor=1.6)

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
    
    def preprocess_val(self, image_path, label=None, ers=False):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        img = tf.image.resize_with_pad(img, 224, 224)

        if ers:
            img = self.fisheye_tf(img, zoom_factor=1.6)

        img = (img - 0.5) * 2.0

        if label is None:
            return img
        return img, label

    def fisheye_tf(self, image, zoom_factor=1.0):
        """
        Applies a fisheye/capsule-style effect to a [H, W, 3] tensor image.
        image: float32 tensor [0,1] or [-1,1] (we will normalize inside)
        zoom_factor: float, zoom in/out
        Returns: [H, W, 3] tensor
        """
        img_shape = tf.shape(image)
        H = tf.cast(img_shape[0],tf.float32)
        W = tf.cast(img_shape[1],tf.float32)

        y, x = tf.meshgrid(tf.range(H), tf.range(W), indexing='ij')
        y = tf.cast(y, tf.float32)
        x = tf.cast(x, tf.float32)

        cx = W / 2.0
        cy = H / 2.0
        nx = (x - cx) / cx
        ny = (y - cy) / cy

        r = tf.sqrt(nx**2 + ny**2)
        theta = tf.atan2(ny, nx)

        k1 = -0.4
        k2 = 0.1
        r_distorted = r * (1 + k1 * r**2 + k2 * r**4)

        x_new = r_distorted * tf.cos(theta)
        y_new = r_distorted * tf.sin(theta)

        x_new /= zoom_factor
        y_new /= zoom_factor

        x_new = (x_new + 1) * cx
        y_new = (y_new + 1) * cy

        coords = tf.stack([y_new, x_new], axis=-1)

        coords = tf.clip_by_value(coords, 0, tf.cast(tf.stack([H-1.0, W-1.0]), tf.float32))

        fisheye_img = tf.expand_dims(image, 0)
        fisheye_img = tf.reshape(tf.image.resize(fisheye_img, [H, W]), [H, W, 3])

        mask_radius = 0.95
        r_mask = tf.sqrt(nx**2 + ny**2)
        mask = tf.cast(r_mask <= mask_radius, tf.float32)
        mask = tf.expand_dims(mask, -1)
        fisheye_img = fisheye_img * mask

        return fisheye_img

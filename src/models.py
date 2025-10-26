import tensorflow as tf
import numpy as np

from tensorflow.keras import layers, models # type: ignore
from src.classes import Classes
from src.dataset_loader import DatasetLoader

class Models:

    def __init__(self, args):
        self.args = args
        self.mode = args.type_num
        self.epochs = args.epochs
        self.k = args.k_folds
        self.model_size = args.model_size
        self.verbose = args.verbose

    def train(self):

        if self.mode == 0:
            self.baseline()
        elif self.mode == 1:
            pass # train on ers test on galar
        elif self.mode == 2:
            pass # train on galar test on ers

    def get_backbone(self):
        if self.model_size == 0:
            base_model = tf.keras.applications.MobileNetV2(
                include_top=False,
                weights="imagenet",
                input_shape=(224, 224, 3),
                pooling="avg"
            )
        elif self.model_size == 1:
            base_model = tf.keras.applications.EfficientNetB0(
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
        fold_results = []

        kf_generator = dataset_loader.split_patients_kfold(
            self.ers_labeled_all, 
            self.ers_labels_all, 
            self.ers_unlabeled_all, 
            self.k
            )

        for fold, (ers_labeled_train, ers_labels_train, ers_labeled_test, ers_labels_test, ers_unlabeled) in enumerate(kf_generator, 1):
            print(f"\n================ Fold {fold}/{self.k} ================")
            print(f"Train samples: {len(ers_labeled_train)} | Test samples: {len(ers_labeled_test)}")

            dataset_labeled_train = tf.data.Dataset.from_tensor_slices((ers_labeled_train, ers_labels_train))
            dataset_labeled_train = dataset_labeled_train.map(self.preprocess_with_padding).batch(8).shuffle(100)

            dataset_labeled_test = tf.data.Dataset.from_tensor_slices((ers_labeled_test, ers_labels_test))
            dataset_labeled_test = dataset_labeled_test.map(self.preprocess_with_padding).batch(8)

            # --- define model ---
            num_classes = Classes().num_classes

            base_model = self.get_backbone()

            model = models.Sequential([
                base_model,
                layers.Dense(num_classes, activation="sigmoid")
            ])

            model.compile(
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"]
            )

            # --- supervised pre-training ---
            model.fit(dataset_labeled_train, epochs=self.epochs, verbose=self.verbose)

            # --- pseudo-label generation ---
            dataset_unlabeled = tf.data.Dataset.from_tensor_slices(ers_unlabeled)
            dataset_unlabeled = dataset_unlabeled.map(self.preprocess_with_padding).batch(8)
            preds = model.predict(dataset_unlabeled, verbose=self.verbose)
            pseudo_labels = (preds > 0.9).astype(np.float32)

            mask_confident = np.max(preds, axis=1) > 0.9
            ers_confident_images = np.array(ers_unlabeled)[mask_confident]
            ers_confident_labels = pseudo_labels[mask_confident]

            # --- retrain with pseudo-labeled data ---
            X_combined = np.concatenate([ers_labeled_train, ers_confident_images])
            Y_combined = np.concatenate([ers_labels_train, ers_confident_labels])

            dataset_combined = tf.data.Dataset.from_tensor_slices((X_combined, Y_combined))
            dataset_combined = dataset_combined.map(self.preprocess_with_padding).batch(8).shuffle(200)
            model.fit(dataset_combined, epochs=self.epochs, verbose=self.verbose)

            # --- evaluate ---
            results = model.evaluate(dataset_labeled_test, verbose=self.verbose)
            print(f"Fold {fold} | Loss: {results[0]:.4f} | AUC: {results[1]:.4f} | Acc: {results[2]:.4f}")
            fold_results.append(results)

            save_path = f"weights/baseline_{self.model_size}_fold{fold}.h5"
            model.save(save_path)
            print(f"Saved {self.model_size} model for fold {fold} -> {save_path}")

        fold_results = np.array(fold_results)
        mean_loss, mean_auc, mean_acc = fold_results.mean(axis=0)
        std_loss, std_auc, std_acc = fold_results.std(axis=0)

        print(f"\n===== {self.k}-Fold Cross-Validation Results =====")
        print(f"Loss: {mean_loss:.4f} ± {std_loss:.4f}")
        print(f"AUC:  {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"Acc:  {mean_acc:.4f} ± {std_acc:.4f}")
        


    
    def preprocess_with_padding(self, image_path, label=None):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)

        img = tf.image.resize_with_pad(img, 224, 224)

        img = tf.cast(img, tf.float32) / 255.0

        if label is None:
            return img
        return img, label
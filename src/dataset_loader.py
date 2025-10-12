import pandas as pd
import json
import numpy as np
import os

from classes import Classes

class DatasetLoader:

    def __init__(self,args):

        self.ers_path = args.ers_path
        self.ers_path = "/mnt/e/ERS/ers_jpg/"
        self.galar_path = '/mnt/e/galar/'

        self.train_size = args.train_size
        self.test_size = args.test_size

        self.type_num = args.type_num


    def prepare_data(self):
        if self.type_num == 0:
            self.prepare_ers()

    def prepare_ers(self):
        classes = Classes()
        base_dir = self.ers_path
        labels_path = f"{base_dir}labels_jpg.csv"

        df = pd.read_csv(labels_path, sep="\t", low_memory=False)

        multi_labels = (
            df.dropna(subset=["label"])
            .groupby("file")["label"]
            .apply(list)
            .reset_index()
        )
        
        class_to_idx = {cls: i for i, cls in enumerate(classes.unified_classes)}
        num_classes = len(class_to_idx)

        def map_labels_to_unified(label_list):
            mapped = set()
            for code in label_list:
                if code in classes.ers_to_galar:
                    mapped.add(classes.ers_to_galar[code])

            return list(mapped)


        def make_multihot(unified_list):
            vec = np.zeros(num_classes, dtype=np.float32)
            for lbl in unified_list:
                if lbl in class_to_idx:
                    vec[class_to_idx[lbl]] = 1.0
            return vec

        multi_labels["unified_labels"] = multi_labels["label"].apply(map_labels_to_unified)
        multi_labels["multi_hot_unified"] = multi_labels["unified_labels"].apply(make_multihot)

        labeled_image_paths = [base_dir + f for f in multi_labels["file"].tolist()]
        multi_hot_labels = np.stack(multi_labels["multi_hot_unified"].values)

        sums = multi_hot_labels.sum(axis=1)

        multi_labels = multi_labels[sums > 0].reset_index(drop=True)

        labeled_image_paths = [base_dir + f for f in multi_labels["file"].tolist()]
        multi_hot_labels = np.stack(multi_labels["multi_hot_unified"].values)

        all_files = df["file"].unique()
        all_file_paths = [base_dir + f for f in all_files]

        labeled_set = set(labeled_image_paths)
        unlabeled_image_paths = [f for f in all_file_paths if f not in labeled_set]

        unlabeled_image_paths = list(np.unique(unlabeled_image_paths))

        return labeled_image_paths, multi_hot_labels, unlabeled_image_paths
    
    def prepare_galar(self):
        classes = Classes()
        base_dir = self.galar_path  # e.g., "datasets/galar/"
        labels_dir = os.path.join(base_dir, "labels")

        class_to_idx = {cls: i for i, cls in enumerate(classes.unified_classes)}
        num_classes = len(class_to_idx)

        labeled_image_paths = []
        multi_hot_labels = []
        unlabeled_image_paths = []

        # Iterate over folders 1–80 (or as many label files as exist)
        for folder_id in range(1, 81):
            folder_path = os.path.join(base_dir, str(folder_id))
            label_path = os.path.join(labels_dir, f"{folder_id}.csv")

            if not os.path.exists(label_path):
                continue
            if not os.path.exists(folder_path):
                continue

            df = pd.read_csv(label_path, sep=",", low_memory=False)

            # Identify columns with binary disease indicators
            binary_cols = [
                c for c in df.columns
                if c not in ["index", "section", "frame"]
            ]

            # Iterate through each row in the label file
            for _, row in df.iterrows():
                frame_number = int(row["frame"])
                frame_file = f"frame_{frame_number:06d}.jpg"
                img_path = os.path.join(folder_path, frame_file)

                if not os.path.exists(img_path):
                    continue  # skip missing files

                # Collect active Galar labels (value == 1)
                active_labels = [col for col in binary_cols if row[col] == 1]

                # Map Galar → unified labels
                unified_labels = []
                for lbl in active_labels:
                    if hasattr(classes, "galar_to_ers") and lbl in classes.galar_to_ers:
                        unified_labels.append(classes.galar_to_ers[lbl])
                    elif lbl in class_to_idx:  # already unified label name
                        unified_labels.append(lbl)

                # If no label active, assign 'healthy'
                if not unified_labels:
                    unified_labels = ["healthy"]

                # Create multi-hot vector
                vec = np.zeros(num_classes, dtype=np.float32)
                for lbl in unified_labels:
                    if lbl in class_to_idx:
                        vec[class_to_idx[lbl]] = 1.0

                labeled_image_paths.append(img_path)
                multi_hot_labels.append(vec)

            # Detect unlabeled images (exist in folder but not in CSV)
            all_imgs = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".png"))
            ]
            labeled_imgs = set(f"frame_{int(fr):06d}.jpg" for fr in df["frame"])
            unlabeled = [os.path.join(folder_path, f) for f in all_imgs if f not in labeled_imgs]
            unlabeled_image_paths.extend(unlabeled)

        # Convert label list → array
        if len(multi_hot_labels) > 0:
            multi_hot_labels = np.stack(multi_hot_labels)
        else:
            multi_hot_labels = np.zeros((0, num_classes), dtype=np.float32)

        unlabeled_image_paths = list(np.unique(unlabeled_image_paths))

        return labeled_image_paths, multi_hot_labels, unlabeled_image_paths

    
loader = DatasetLoader().prepare_galar()

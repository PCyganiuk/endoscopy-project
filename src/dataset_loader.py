import pandas as pd
import json
import numpy as np
import os

from sklearn.model_selection import train_test_split
import re

from classes import Classes

class DatasetLoader:

    def __init__(self,):

        #self.ers_path = args.ers_path
        self.ers_path = "/mnt/e/ERS/ers_jpg/"
        self.galar_path = '/mnt/e/galar/'

        #self.train_size = args.train_size
        #self.test_size = args.test_size

        #self.type_num = args.type_num


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

        output_path = f"{base_dir}ers_dataset_summary.txt"
        with open(output_path, "w") as f:
            # labeled section
            for path, label_vec in zip(labeled_image_paths, multi_hot_labels):
                label_str = " ".join(map(str, label_vec.astype(int)))
                f.write(f"{path} {label_str}\n")

            # unlabeled section
            f.write("\n# Unlabeled images\n")
            for path in unlabeled_image_paths:
                f.write(f"{path}\n")

        print(f"Saved dataset summary to {output_path}")

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
                frame_file = f"frame_{frame_number:06d}.PNG"
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

        output_path = f"{base_dir}ers_dataset_summary.txt"
        with open(output_path, "w") as f:
            # labeled section
            for path, label_vec in zip(labeled_image_paths, multi_hot_labels):
                label_str = " ".join(map(str, label_vec.astype(int)))
                f.write(f"{path} {label_str}\n")

            # unlabeled section
            f.write("\n# Unlabeled images\n")
            for path in unlabeled_image_paths:
                f.write(f"{path}\n")

        print(f"Saved dataset summary to {output_path}")

        return labeled_image_paths, multi_hot_labels, unlabeled_image_paths
    
    def split_by_patient_id(
        self,
        labeled_image_paths: list[str],
        multi_hot_labels: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42
        ):


        # Extract patient IDs (e.g., /0001/samples/ → patient_id = "0001")
        def extract_patient_id(path: str):
            match = re.search(r"/(\d+)/", path)
            return match.group(1) if match else None

        patient_ids = [extract_patient_id(p) for p in labeled_image_paths]
        patient_ids = np.array(patient_ids)

        # Identify unique patients and split
        unique_patients = np.unique([pid for pid in patient_ids if pid is not None])
        train_patients, test_patients = train_test_split(
            unique_patients, test_size=test_size, random_state=random_state
        )

        # Create masks
        train_mask = np.isin(patient_ids, train_patients)
        test_mask = np.isin(patient_ids, test_patients)

        # Split labeled data
        labeled_image_paths_train = np.array(labeled_image_paths)[train_mask].tolist()
        multi_hot_labels_train = multi_hot_labels[train_mask]

        labeled_image_paths_test = np.array(labeled_image_paths)[test_mask].tolist()
        multi_hot_labels_test = multi_hot_labels[test_mask]

        print(f"Train patients: {len(train_patients)} | Test patients: {len(test_patients)}")
        print(f"Train samples: {len(labeled_image_paths_train)} | Test samples: {len(labeled_image_paths_test)}")

        return (
            labeled_image_paths_train,
            multi_hot_labels_train,
            labeled_image_paths_test,
            multi_hot_labels_test,
        )


    
labeled, label, unalebed =  DatasetLoader().prepare_ers()
DatasetLoader().split_by_patient_id(labeled, label, unalebed)


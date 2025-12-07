import pandas as pd
import numpy as np
import os
import re

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold

from src.classes import Classes

class DatasetLoader:

    def __init__(self, args):

        self.ers_path = args.ers_path
        self.galar_path = args.galar_path
        #self.ers_path = "/mnt/d/ERS/ers_jpg/"
        #self.galar_path = '/mnt/e/galar_jpg/'

        self.train_size = args.train_size
        self.test_size = args.test_size
        self.binary = args.binary
        self.type_num = args.type_num

    def prepare_ers(self):
        classes = Classes(self.binary)
        base_dir = self.ers_path
        labels_path = f"{base_dir}labels.csv"

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

        if classes.binary:
            unhealthy_indices = np.where(np.array(classes.unified_classes) != "healthy")[0]
            healthy_idx = classes.unified_classes.index("healthy")

            binary_labels = np.zeros((multi_hot_labels.shape[0], 2), dtype=np.float32)

            has_unhealthy = (multi_hot_labels[:, unhealthy_indices].sum(axis=1) > 0).astype(np.float32)
            has_healthy = multi_hot_labels[:, healthy_idx]

            binary_labels[:, 0] = has_unhealthy
            binary_labels[:, 1] = np.logical_and(has_healthy, np.logical_not(has_unhealthy)).astype(np.float32)

            multi_hot_labels = binary_labels



        output_path = f"data_summary/ers_dataset_summary.txt"
        with open(output_path, "w") as f:
            for path, label_vec in zip(labeled_image_paths, multi_hot_labels):
                label_str = " ".join(map(str, label_vec.astype(int)))
                f.write(f"{path} {label_str}\n")

            f.write("\n# Unlabeled images\n")
            for path in unlabeled_image_paths:
                f.write(f"{path}\n")

        print(f"Saved dataset summary to {output_path}")

        return labeled_image_paths, multi_hot_labels, unlabeled_image_paths
    
    def prepare_galar(self):
        classes = Classes(False)
        base_dir = self.galar_path
        labels_dir = os.path.join(base_dir, "Labels")

        class_to_idx = {cls: i for i, cls in enumerate(classes.unified_classes)}
        num_classes = len(class_to_idx)

        labeled_image_paths = []
        multi_hot_labels = []

        for folder_id in range(1, 81):
            folder_path = os.path.join(base_dir, str(folder_id))
            label_path = os.path.join(labels_dir, f"{folder_id}.csv")

            if not os.path.exists(label_path) or not os.path.exists(folder_path):
                continue

            df = pd.read_csv(label_path, sep=",", low_memory=False)

            binary_cols = [c for c in df.columns if c not in ["index", "section", "frame"]]

            labeled_set = set()
            for _, row in df.iterrows():
                if str(row.get("section", "")).lower() in ["mouth", "esophagus"]:
                    continue

                frame_number = int(row["frame"])
                frame_file = f"frame_{frame_number:06d}.jpg"
                img_path = os.path.join(folder_path, frame_file)

                if not os.path.exists(img_path):
                    continue

                active_labels = [col for col in binary_cols if row[col] == 1]

                unified_labels = []
                for lbl in active_labels:
                    if hasattr(classes, "galar_to_ers") and lbl in classes.galar_to_ers:
                        unified_labels.append(classes.galar_to_ers[lbl])
                    elif lbl in class_to_idx:
                        unified_labels.append(lbl)

                if not unified_labels:
                    unified_labels = ["healthy"]

                vec = np.zeros(num_classes, dtype=np.float32)
                for lbl in unified_labels:
                    if lbl in class_to_idx:
                        vec[class_to_idx[lbl]] = 1.0

                labeled_image_paths.append(img_path)
                multi_hot_labels.append(vec)
                labeled_set.add(frame_file.lower())

            all_imgs = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png"))]
            for img_file in all_imgs:
                if img_file.lower() not in labeled_set:
                    img_path = os.path.join(folder_path, img_file)
                    labeled_image_paths.append(img_path)
                    multi_hot_labels.append(np.zeros(num_classes, dtype=np.float32))

        multi_hot_labels = np.stack(multi_hot_labels)

        if self.binary:
            unhealthy = (multi_hot_labels[:, :16].sum(axis=1) > 0).astype(np.float32)
            healthy = multi_hot_labels[:, 16].astype(np.float32)

            healthy = np.logical_and(healthy, np.logical_not(unhealthy)).astype(np.float32)

            multi_hot_labels = np.stack([unhealthy, healthy], axis=1).astype(np.float32)




        output_path = f"data_summary/galar_dataset_summary.txt"
        with open(output_path, "w") as f:
            for path, label_vec in zip(labeled_image_paths, multi_hot_labels):
                label_str = " ".join(map(str, label_vec.astype(int)))
                f.write(f"{path} {label_str}\n")

        print(f"Saved dataset summary to {output_path}")

        return labeled_image_paths, multi_hot_labels
    
    def prepare_kvasir(self):
        classes = Classes(False)
        labeled_image_paths = []
        multi_hot_labels = []
        for class_name, label in classes.kvasir_label_map.items():
            class_path = os.path.join(self.kvasir_path, class_name)
            if not os.path.exists(class_path):
                continue
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(class_path, fname)
                    labeled_image_paths.append(file_path)
                    multi_hot_labels.append(label)
        
        return labeled_image_paths, multi_hot_labels

    
    def split_labeled_by_patient_id(
        self,
        labeled_image_paths: list[str],
        multi_hot_labels: np.ndarray,
        unlabeled_image_paths: list[str],
        test_size: float = 0.2,
        random_state: int = 42
        ):

        def extract_patient_id(path: str):
            match = re.search(r"/(\d+)/", path)
            return match.group(1) if match else None

        patient_ids = [extract_patient_id(p) for p in labeled_image_paths]
        patient_ids = np.array(patient_ids)

        unique_patients = np.unique([pid for pid in patient_ids if pid is not None])
        
        train_patients, test_patients = train_test_split(
            unique_patients, test_size=test_size
        )

        train_mask = np.isin(patient_ids, train_patients)
        test_mask = np.isin(patient_ids, test_patients)

        labeled_image_paths_train = np.array(labeled_image_paths)[train_mask].tolist()
        multi_hot_labels_train = multi_hot_labels[train_mask]

        labeled_image_paths_test = np.array(labeled_image_paths)[test_mask].tolist()
        multi_hot_labels_test = multi_hot_labels[test_mask]

        unlabeled_patient_ids = [extract_patient_id(p) for p in unlabeled_image_paths]
        unlabeled_mask = np.isin(unlabeled_patient_ids, train_patients)
        unlabeled_image_paths_filtered = np.array(unlabeled_image_paths)[unlabeled_mask].tolist()

        print(f"Train patients: {len(train_patients)} | Test patients: {len(test_patients)}")
        print(f"Train samples: {len(labeled_image_paths_train)} | Test samples: {len(labeled_image_paths_test)}")
        print(f"Filtered unlabeled samples: {len(unlabeled_image_paths_filtered)} / {len(unlabeled_image_paths)}")
        
        return (
            labeled_image_paths_train,
            multi_hot_labels_train,
            labeled_image_paths_test,
            multi_hot_labels_test,
            unlabeled_image_paths_filtered,
        )

    def split_patients_kfold(
        self,
        labeled_image_paths: list[str],
        labels: np.array,
        unlabeled_image_paths: list[str] = None,
        n_splits=20
    ):
        def extract_patient_id(path):
            match = re.search(r"/(\d+)/", path)
            return match.group(1) if match else None

        patient_ids = np.array([extract_patient_id(p) for p in labeled_image_paths])

        y = labels.argmax(axis=1)

        X_dummy = np.zeros(len(y))
        groups = patient_ids

        sgkf = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True, random_state=14
        )

        for fold, (train_idx, test_idx) in enumerate(
            sgkf.split(X_dummy, y, groups=groups)
        ):
            train_mask = train_idx
            test_mask = test_idx

            labeled_train = np.array(labeled_image_paths)[train_mask].tolist()
            labels_train = labels[train_mask]

            labeled_test = np.array(labeled_image_paths)[test_mask].tolist()
            labels_test = labels[test_mask]

            if unlabeled_image_paths is not None:
                unlabeled_patient_ids = np.array([extract_patient_id(p) for p in unlabeled_image_paths])
                train_patients = np.unique(groups[train_idx])
                unlabeled_mask = np.isin(unlabeled_patient_ids, train_patients)
                unlabeled_filtered = np.array(unlabeled_image_paths)[unlabeled_mask].tolist()
            else:
                unlabeled_filtered = []

            print(
                f"Fold {fold + 1}/{n_splits}: "
                f"Train samples={len(train_idx)}, Test samples={len(test_idx)}, "
                f"Train patients={len(np.unique(groups[train_idx]))}, "
                f"Test patients={len(np.unique(groups[test_idx]))}"
            )

            yield labeled_train, labels_train, labeled_test, labels_test, unlabeled_filtered



#loader = DatasetLoader()
#labeled,labels = loader.prepare_galar()
#loader.split_labeled_by_patient_id(labeled, labels)

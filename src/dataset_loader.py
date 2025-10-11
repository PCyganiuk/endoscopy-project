import pandas as pd
import json
import numpy as np

from classes import Classes

class DatasetLoader:

    def __init__(self,args):

        self.ers_path = args.ers_path
        self.ers_path = "/mnt/e/ERS/ers_jpg/"
        self.galar_path = args.galar_path

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

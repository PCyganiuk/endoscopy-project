def count_classes(file_path: str, num_classes: int = 17):
    """
    Counts how many samples belong to each of the 17 classes.
    Assumes each line is formatted as:
    <image_path> <label1> <label2> ... <label17>
    """
    class_counts = [0] * num_classes
    total_lines = 0
    multi_label_samples = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != num_classes + 1:
                continue

            labels = list(map(int, parts[1:]))
            total_lines += 1

            for i, val in enumerate(labels):
                if val == 1:
                    class_counts[i] += 1

            if sum(labels) > 1:
                multi_label_samples += 1

    print(f"Total samples: {total_lines}")
    for i, count in enumerate(class_counts):
        print(f"Class {i}: {count}")
    print(f"Multi-label samples: {multi_label_samples}")


if __name__ == "__main__":
    dataset_file = "/home/pyza/Projects/endo-project/data_summary/galar_dataset_summary.txt"
    count_classes(dataset_file)

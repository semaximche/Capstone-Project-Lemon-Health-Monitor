import os
import csv
import tensorflow as tf
import keras as keras
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import random
import shutil
import squarify
import cv2
import matplotlib.image as mpimg
import seaborn as sns
import sklearn
import ultralytics
from sklearn.model_selection import train_test_split
from ultralytics import YOLO


def csv_to_dataframe(csv_filepath, columns_name):
    df = pd.read_csv(csv_filepath, names=columns_name, header=0)
    return df


def calc_percentage(df_total, df_part, file_id_column):
    part = len(df_part[file_id_column].unique())
    total = len(df_total[file_id_column].unique())
    return (part / total) * 100


def dict_to_yaml(data):
    yaml_lines = []
    for key, value in data.items():
        if key == 'format':
            if value == 'line_break':
                yaml_lines.append('')
        elif isinstance(value, list):
            yaml_lines.append(f"{key}: {value}")
        else:
            yaml_lines.append(f"{key}: {value}")
    return '\n'.join(yaml_lines)


def create_yaml_data(yaml_file_name, output_dir, data):
    yaml_content = dict_to_yaml(data)
    output_path = os.path.join(output_dir, f'{yaml_file_name}.yaml')
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as file:
        file.write(yaml_content)
    print(f'YAML file created successfully in {output_path}.')


def convert_bbox_to_yolo(x, y, width, height, img_width, img_height):
    x_center = (x + width / 2) / img_width
    y_center = (y + height / 2) / img_height
    width /= img_width
    height /= img_height
    return x_center, y_center, width, height


def dataframe_yolofiles(df, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data = {}
    for index, row in df.iterrows():
        image_id = row['image_id']
        img_width = int(row['width'])
        img_height = int(row['height'])
        bbox = eval(row['bbox'])
        x, y, width, height = bbox
        x_center, y_center, norm_width, norm_height = convert_bbox_to_yolo(x, y, width, height, img_width, img_height)
        if image_id not in data:
            data[image_id] = []
        data[image_id].append(f"0 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
    for image_id, bboxes in data.items():
        output_file_path = os.path.join(output_dir, os.path.splitext(image_id)[0] + ".txt")
        with open(output_file_path, 'w') as f:
            for bbox in bboxes:
                f.write(bbox + "\n")
    print(f'Total text files created in: {output_dir}')


def move_images_to_directory(df, root_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    unique_image_ids = df['image_id'].unique()
    for image_id in unique_image_ids:
        source_path = os.path.join(root_dir, image_id)
        target_path = os.path.join(target_dir, image_id)
        if os.path.exists(source_path):
            shutil.copy(source_path, target_path)
        else:
            print(f"Warning: Image {image_id} not found on path {source_path}")
    print(f'Total images moved to: {target_dir}')


def leaf_detect(img_path, model):
    img = cv2.imread(img_path)
    detect_result = model(img)
    detect_img = detect_result[0].plot()
    detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
    return detect_img


def plot_images_with_detections(images_dir, labels_dir, num_images=16, grid_size=(4, 4), figsize=(16, 16)):
    image_files = os.listdir(images_dir)
    image_files.sort()
    images_forshow = image_files[:num_images]
    fig, axs = plt.subplots(grid_size[0], grid_size[1], figsize=figsize)
    for i, image_file in enumerate(images_forshow):
        row = i // grid_size[1]
        col = i % grid_size[1]
        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path)
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, "r") as f:
            labels = f.read().strip().split("\n")
        for label in labels:
            if len(label.split()) != 5:
                continue
            class_id, x_center, y_center, width, height = map(float, label.split())
            x_min = int((x_center - width / 2) * image.shape[1])
            y_min = int((y_center - height / 2) * image.shape[0])
            x_max = int((x_center + width / 2) * image.shape[1])
            y_max = int((y_center + height / 2) * image.shape[0])
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (250, 221, 47), 3)
        axs[row, col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[row, col].axis('off')
    plt.show()


def show_csv_results(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(15, 15))
    sns.lineplot(x='epoch', y='train/box_loss', data=df, ax=axs[0, 0])
    sns.lineplot(x='epoch', y='train/cls_loss', data=df, ax=axs[0, 1])
    sns.lineplot(x='epoch', y='train/dfl_loss', data=df, ax=axs[1, 0])
    sns.lineplot(x='epoch', y='metrics/precision(B)', data=df, ax=axs[1, 1])
    sns.lineplot(x='epoch', y='metrics/recall(B)', data=df, ax=axs[2, 0])
    sns.lineplot(x='epoch', y='metrics/mAP50(B)', data=df, ax=axs[2, 1])
    sns.lineplot(x='epoch', y='metrics/mAP50-95(B)', data=df, ax=axs[3, 0])
    sns.lineplot(x='epoch', y='val/box_loss', data=df, ax=axs[3, 1])
    sns.lineplot(x='epoch', y='val/cls_loss', data=df, ax=axs[4, 0])
    sns.lineplot(x='epoch', y='val/dfl_loss', data=df, ax=axs[4, 1])
    axs[0, 0].set(title='Train Box Loss')
    axs[0, 1].set(title='Train Class Loss')
    axs[1, 0].set(title='Train DFL Loss')
    axs[1, 1].set(title='Metrics Precision (B)')
    axs[2, 0].set(title='Metrics Recall (B)')
    axs[2, 1].set(title='Metrics mAP50 (B)')
    axs[3, 0].set(title='Metrics mAP50-95 (B)')
    axs[3, 1].set(title='Validation Box Loss')
    axs[4, 0].set(title='Validation Class Loss')
    axs[4, 1].set(title='Validation DFL Loss')
    plt.suptitle('Training Metrics and Loss', fontsize=24)
    plt.subplots_adjust(top=0.8)
    plt.tight_layout()
    plt.show()


def show_directory_images(directory, num_images, rows=3, columns=3):
    archivos_imagen = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    fig = plt.figure(figsize=(10, 10))
    for i in range(min(num_images, len(archivos_imagen))):
        img_path = os.path.join(directory, archivos_imagen[i])
        img = cv2.imread(img_path)
        ax = fig.add_subplot(rows, columns, i + 1)
        ax.imshow(img)
        ax.axis('off')
    for j in range(num_images, rows * columns):
        ax = fig.add_subplot(rows, columns, j + 1)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def show_metrics(metrics):
    # Create the barplot
    ax = sns.barplot(x=['mAP50-95', 'mAP50', 'mAP75'], y=[metrics.box.map, metrics.box.map50, metrics.box.map75])
    # Set the title and axis labels
    ax.set_title('YOLO Evaluation Metrics')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    # Set the figure size
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    # Add the values on top of the bars
    for p in ax.patches:
        ax.annotate('{:.3f}'.format(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), ha='center',
                    va='bottom')
    # Show the plot
    plt.show()


def show_detections(images_dir, num_images=16, rows=4, columns=4):
    image_files = os.listdir(images_dir)
    selected_images = random.sample(image_files, num_images)
    fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(15, 15))
    for i, img_file in enumerate(selected_images):
        row_idx = i // columns
        col_idx = i % columns
        img_path = os.path.join(images_dir, img_file)
        detect_img = leaf_detect(img_path, model)
        axes[row_idx, col_idx].imshow(detect_img)
        axes[row_idx, col_idx].axis('off')
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()


def show_image(img_path):
    img = mpimg.imread(img_path)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(img)
    ax.axis('off')
    plt.show()


if __name__ == '__main__':
    random_seed = 42
    random.seed(random_seed)

    # CSV to Yolo format
    columns_name = ['image_id', 'width', 'height', 'bbox']
    csv_filepath = './LeafDetection/train.csv'
    annotations = pd.read_csv(csv_filepath, names=columns_name, header=0)
    df = pd.DataFrame(annotations)

    # Split the DataFrame into training(80%) and validation with test(20%) based on the unique file names
    unique_filenames = df['image_id'].unique()
    train_filenames, val_test_filenames = train_test_split(unique_filenames, test_size=0.2, random_state=random_seed)
    train_df = df[df['image_id'].isin(train_filenames)]

    # 10% for validation and 10% for test
    val_filenames, test_filenames = train_test_split(val_test_filenames, test_size=0.5, random_state=random_seed)
    val_df = df[df['image_id'].isin(val_filenames)]
    test_df = df[df['image_id'].isin(test_filenames)]

    # Show elements in each set

    train_percentage = calc_percentage(df, train_df, 'image_id')
    print(f'Elements in train_df({train_percentage}%):', len(train_df['image_id'].unique()))
    train_df.head()

    val_percentage = calc_percentage(df, val_df, 'image_id')
    print(f'Elements in val_df({val_percentage}%):', len(val_df['image_id'].unique()))
    val_df.head()

    test_percentage = calc_percentage(df, test_df, 'image_id')
    print(f'Elements in test_df({test_percentage}%):', len(test_df['image_id'].unique()))
    test_df.head()

    # Loading a pretrained model
    model = YOLO('yolo11n.pt')

    # Training the model
    model.train(data='./LeafDetection/yolo_dataset_leafdetection/data.yaml',
                epochs=60,
                imgsz=1024,
                device='cuda',
                batch=-1,
                workers=3,
                )

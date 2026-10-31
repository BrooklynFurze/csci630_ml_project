import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

def clean_data(df, image_dir):
    # Count missing values per column
    # There are none, but just to demonstrate for posterity
    print("Null values: img_id:", df['img_id'].isnull().sum(), ", target:", df['target'].isnull().sum())

    #Check to make sure all the images correspond to real files
    df["image_path"] = df["img_id"].apply(
        lambda name: image_dir / name
    )
    df["exists"] = df["image_path"].apply(
        lambda path: path.is_file()
    )
    missing_files = df["exists"] == False
    print("Missing image files:", missing_files.sum())

    #Check for duplicate rows, there are none
    print("Duplicate rows:", df.duplicated().sum())

    # Check for any rows with duplicate filenames, there are none
    print("Duplicate filename rows:",
        df["img_id"].duplicated().sum())

    #Examining for class imbalance:
    #This shows that there are exactly 55 images of each class in the training dataset
    class_counts = (
        df["target"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    class_counts.columns = [
        "Class ID Num",
        "Number of Samples in Class"
    ]
    print(class_counts.to_string(index=False))

def load_image_stretch(filename, image_dir, dimensions):
    image_path = image_dir / str(filename)

    image = Image.open(image_path).convert('RGB')
    size = (dimensions, dimensions)
    image = image.resize(size)

    return np.array(image)

def load_image_pad(filename, image_dir, dimensions):
    image_path = image_dir / str(filename)

    image = Image.open(image_path).convert('RGB')
    size = (dimensions, dimensions)
    image.thumbnail(size)
    new_image = Image.new('RGB', size, (0, 0, 0))

    x_offset = (size[0] - image.width) // 2
    y_offset = (size[1] - image.height) // 2

    new_image.paste(image, (x_offset, y_offset))
    return np.array(new_image)

def load_image_dataset(filenames, image_dir, dimensions, variant):
    images = []

    if (variant == "pad"):
        for filename in filenames:
            image = load_image_pad(filename, image_dir, dimensions)
            images.append(image.flatten())
    elif (variant == "stretch"): 
        for filename in filenames:
            image = load_image_stretch(filename, image_dir, dimensions)
            images.append(image.flatten())
    

    return np.array(images)


def main():
    # data setup
    project_dir = Path(__file__).resolve().parent.parent
    csv_path = project_dir / "data" / "train.csv"
    data_path = project_dir / "data" / "train"
    image_dir = Path(data_path)
    df = pd.read_csv(csv_path)

    #data cleaning
    #Looking at the results of this it looks like there isn't actually any cleaning needed
    #E.g. no null values, classes are perfectly balanced, all image files correspond to real files
    #So I'm opting to leave it in to demonstrate that the data is already clean
    print("Checking and cleaning the data: ")
    clean_data(df, image_dir)
    print("---------------------------------------------------\n\n")

    #splitting into x (image ids) and y (target)
    x = df['img_id']
    y = df['target']

    #Originally thought that a test dataset wouldn't be needed because they've already 
    #split the images into two folders. However, the second folder isn't labeled, and 
    #attempts to label all of them have been difficult on my part. 
    #So I'm just going to pull a 15% test split off of the training dataset 
    #Theoretically I could bring the images that I was able to successfully label into the 
    #training dataset but this would introduce class imbalance
    #as I am far better at identifying certain species of birds than others
    #Also assuming random distribution on a perfectly distributed dataset this also be roughly perfectly distributed
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=35, stratify=y)

    print("Training the model... (takes a bit)")

    # overall goal is to compare and contrast different methods
    #so the resizing will have two variants: stretching (distorts image) and padding (doesn't distort image)
    #and however many variations of image dimensions are desired

    x_train_images_stretch = load_image_dataset(x_train, image_dir, 64, "stretch")
    x_test_images_stretch = load_image_dataset(x_test, image_dir, 64, "stretch")
    x_train_images_pad = load_image_dataset(x_train, image_dir, 64, "pad")
    x_test_images_pad = load_image_dataset(x_test, image_dir, 64, "pad")

    #normalize 
    x_train_images_stretch = x_train_images_stretch / 255.0
    x_test_images_stretch = x_test_images_stretch / 255.0
    x_train_images_pad = x_train_images_pad / 255.0
    x_test_images_pad = x_test_images_pad / 255.0

    rf_stretch = RandomForestClassifier(
        n_estimators=100,
        max_depth=50,
        random_state=35,
        n_jobs=-1
    )
    rf_pad = RandomForestClassifier(
        n_estimators=100, 
        max_depth=50, 
        random_state=35, 
        n_jobs=-1
    )

    rf_stretch.fit(x_train_images_stretch, y_train)
    rf_pad.fit(x_train_images_pad, y_train)

    #evaluating
    # 50 classes means on a random guess it should only be getting it right 2% of the time
    y_pred_stretch = rf_stretch.predict(x_test_images_stretch)
    y_pred_pad = rf_pad.predict(x_test_images_pad)
    print("Metrics per model:")

    print("Stretch Accuracy:", accuracy_score(y_test, y_pred_stretch))
    print("Stretch Macro Precision: ", precision_score(y_test, y_pred_stretch, average='macro', zero_division=0))
    print("Stretch Macro Recall: ", recall_score(y_test, y_pred_stretch, average='macro', zero_division=0))
    print("Stretch Macro F1: ", f1_score(y_test, y_pred_stretch, average='macro', zero_division=0))

    print()

    print("Pad Accuracy:", accuracy_score(y_test, y_pred_pad))
    print("Pad Macro Precision: ", precision_score(y_test, y_pred_pad, average='macro', zero_division=0))
    print("Pad Macro Recall: ", recall_score(y_test, y_pred_pad, average='macro', zero_division=0))
    print("Pad Macro F1: ", f1_score(y_test, y_pred_pad, average='macro', zero_division=0))

    #TODO more methods of analyzing the images to try and get better metrics
    #TODO analyze which classes its struggling on for the presentation



if __name__ == '__main__':
    main()
import pandas as pd
from pathlib import Path

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



def main():
    # data setup
    project_dir = Path(__file__).resolve().parent.parent
    csv_path = project_dir / "data" / "train.csv"
    train_path = project_dir / "data" / "train"
    train_image_dir = Path(train_path)
    train_df = pd.read_csv(csv_path)

    #TODO set up training dataset once the issues are ironed out

    #data cleaning
    #Looking at the results of this it looks like there isn't actually any cleaning needed
    #E.g. no null values, classes are perfectly balanced, all image files correspond to real files
    #So I'm opting to leave it in to demonstrate that the data is already clean
    print("Checking and cleaning the data: ")
    clean_data(train_df, train_image_dir)
    print("---------------------------------------------------\n\n")

    #train test split is not needed as they've already split the training set off


    print("Training the model...")




if __name__ == '__main__':
    main()
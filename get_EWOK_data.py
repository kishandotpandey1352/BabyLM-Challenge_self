from datasets import load_dataset
import pandas as pd


def download_and_save_ewok():
    print("Downloading the ewok-core/ewok-core-1.0 dataset...")
    dataset = load_dataset("ewok-core/ewok-core-1.0", token=True)

    for split in dataset:
        df = pd.DataFrame(dataset[split])
        filename = f"ewok_{split}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved {split} split to {filename}")

if __name__ == "__main__":
    download_and_save_ewok()

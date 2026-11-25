from src import cleaning, download_data, features


def main():
    download_data.main()     # latest Nifty and VIX (yfinance)
    cleaning.main()          # merge and validate
    features.main()          # targets and features -> data/processed/model_dataset.csv


if __name__ == "__main__":
    main()
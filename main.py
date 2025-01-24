
import os

def run_preprocessing():
    """
    Run the data preprocessing script.
    """
    print("Running data preprocessing...")
    os.system("python scripts/preprocess.py")
    print("Data preprocessing completed.")

def run_sentiment_analysis():
    """
    Run the sentiment analysis script.
    """
    print("Running sentiment analysis...")
    os.system("python scripts/llm_configuration.py")
    print("Sentiment analysis completed.")

if __name__ == "__main__":
    print("Starting Sentiment Analysis Workflow...")

    # Step 1: Preprocess the data
    run_preprocessing()

    # Step 2: Perform sentiment analysis
    run_sentiment_analysis()

    print("Workflow completed successfully.")



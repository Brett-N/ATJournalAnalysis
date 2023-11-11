import pandas as pd
import datetime
import matplotlib.pyplot as plt

def convert_to_date(date_str):
    if not isinstance(date_str, str):
        return None
    try:
        date_object = datetime.datetime.strptime(date_str, "%b %d, %a")
        return date_object.replace(year=2016)
    except ValueError:
        print(f"Problem with date: {date_str}")
        return None

def visualize_sentiment_over_time(file_path: str):
    df = pd.read_excel(file_path)
    df['Date'] = df['Date'].apply(convert_to_date)
    
    # Set Date as the index for resampling purposes
    df.set_index('Date', inplace=True)

    # Resample by week and compute the mean for each week
    weekly_mean = df['Sentiment'].resample('W').mean()

    # Plotting
    plt.figure(figsize=(15,7))
    plt.plot(weekly_mean.index, weekly_mean.values, marker='o', linestyle='-')
    plt.title('Average Sentiment over Time (Weekly)')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment Score')
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    input_file = input("Enter the path to your Excel file with sentiment data: ")
    visualize_sentiment_over_time(input_file)
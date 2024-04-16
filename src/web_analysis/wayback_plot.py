import argparse
import json
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def read_data(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

def prepare_plot_data(data):
    change_counts = data["change_counts"]
    weeks = list(change_counts.keys())
    counts = list(change_counts.values())
    dates = [datetime.strptime(week + "-1", "%Y-%W-%w") for week in weeks]
    return dates, counts

def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot change counts over time from JSON data.")
    parser.add_argument("data_directory", help="Directory containing JSON files with data.")
    parser.add_argument("output_filename", help="Filename for the output plot PNG file.")
    return parser.parse_args()

def main():
    args = parse_arguments()

    json_files = [os.path.join(args.data_directory, file) for file in os.listdir(args.data_directory) if file.endswith(".json")]
    fig, ax = plt.subplots(figsize=(12, 6))

    for json_file in json_files:
        data = read_data(json_file)
        dates, counts = prepare_plot_data(data)
        ax.plot(dates, counts, marker='o', linestyle='-', linewidth=2, markersize=6, label=os.path.basename(json_file).split('.')[0])

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45, ha='right')

    ax.set_title("Website Change Counts Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of changes (sampled weekly)")

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title="Sites", loc="upper left")

    plt.tight_layout()
    plt.savefig(args.output_filename, dpi=300)
    print(f"Plot saved as '{args.output_filename}'")

if __name__ == "__main__":
    main()

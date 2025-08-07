import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Plot metrics from a study.')
    parser.add_argument('--input_file', type=str, default='overfitting_metrics.csv',
                        help='Path to the input CSV file.')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save the plots.')
    parser.add_argument('--lower_percentile', type=int, default=25,
                        help='Lower percentile for the shaded region.')
    parser.add_argument('--upper_percentile', type=int, default=75,
                        help='Upper percentile for the shaded region.')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Read the data
    try:
        df = pd.read_csv(args.input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_file}")
        return

    metric_groups = {
        'Absolute Errors': ['train_abs_error_E', 'train_abs_error_N'],
        'Losses': ['train_loss', 'train_loss_E', 'train_loss_N', 'train_loss_matrix'],
        'MAE Matrix': ['train_mae_matrix'],
        'Percentage Errors': ['train_percent_E', 'train_percent_N', 'train_percent_matrix']
    }

    for title, metrics in metric_groups.items():
        plt.figure(figsize=(10, 6))
        for metric in metrics:
            if metric not in df.columns:
                print(f"Warning: Metric '{metric}' not found in the data. Skipping.")
                continue

            # Group by epoch and calculate statistics
            grouped = df.groupby('epoch')[metric]
            mean = grouped.mean()
            lower = grouped.quantile(args.lower_percentile / 100)
            upper = grouped.quantile(args.upper_percentile / 100)

            # Plot the mean line and shaded region
            plt.plot(mean.index, mean, label=metric)
            plt.fill_between(mean.index, lower, upper, alpha=0.2)

        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        output_file = os.path.join(args.output_dir, f"{title.replace(' ', '_').lower()}.png")
        plt.savefig(output_file)
        plt.close()
        print(f"Saved plot to {output_file}")

if __name__ == '__main__':
    main()

# Helper functions for moving targets
import pandas as pd
from matplotlib import pyplot as plt


def plot_hold_table(hold_table, vts, distances):
    df = pd.DataFrame(hold_table, columns=vts, index=distances)
    df.index.name = 'd (m)'
    df.columns.name = 'v (km/hob)'
    # Adjust Jupyter notebook display settings for better visibility
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)  # Adjust column width to fit the content
    # Apply styles to make column headers bold
    df = df.style.set_table_styles(
        {'Velocity (m/s)': [{'selector': 'th', 'props': [('font-weight', 'bold')]}]}
    ).format("{:.1f}").set_caption("<h3>Hold Table for Moving Targets</h3>")
    return df

def save_hold_table(df):
    # Create PDF
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))  # You can adjust the size as needed

    # Hide axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    # Ensure df is a DataFrame, not a Styler, before creating the table
    if isinstance(df, pd.io.formats.style.Styler):
        df = df.data
    # Create a table
    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')

    # Add labels
    ax.set_xlabel('Columns')
    ax.set_ylabel('Rows')

    # Add caption
    # Caption at the top of the table
    caption = 'Hold mrads for different target speeds and distances.'
    plt.text(0.5, 0.8, caption, horizontalalignment='center', fontsize=18, transform=ax.transAxes)

    # Add labels for speed (columns) and distance (rows)
    plt.text(0.5, 0.2, 'Speed (km/hob)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
             fontsize=12)
    plt.text(-0.16, 0.5, 'Distance (m)', horizontalalignment='center', verticalalignment='center',
             transform=ax.transAxes, fontsize=12, rotation='vertical')

    # Adjust layout for table font size and scaling
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Save the table as a PDF file
    plt.savefig("hold_table.pdf", format="pdf")
    plt.show()
    plt.close()

    print("PDF file created successfully.")

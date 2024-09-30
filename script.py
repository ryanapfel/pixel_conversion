import click
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import os
import pickle
from tqdm import tqdm


def rgb_to_value(rgb, cmap, min_val=0, max_val=100):
    """
    Maps an RGB color to a numerical value using a colormap.
    """
    # Normalize RGB to [0, 1]
    normalized_rgb = np.array(rgb) / 255.0
    
    # Use the colormap to find the corresponding value
    distances = []
    for i in range(256):
        cmap_color = cmap(i / 255.0)[:3]  # Get the RGB values from the colormap
        dist = np.sqrt(np.sum((cmap_color - normalized_rgb)**2))
        distances.append(dist)
    
    # Find the closest match in the colormap
    idx = np.argmin(distances)
    
    # Normalize index to the range [min_val, max_val]
    normalized_value = idx / 255.0
    value = normalized_value * (max_val - min_val) + min_val
    
    return value

def create_colormap_from_image(image_path):
    """
    Create a colormap by processing pixels from the center downward.
    """
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    center_x = width // 2

    # Extract colors from the center downwards
    colors = []
    for y in range(height):
        rgb = img.getpixel((center_x, y))
        colors.append(tuple(np.array(rgb) / 255.0))  # Normalize to [0, 1]

    # Create a colormap from the extracted colors
    cmap_name = os.path.splitext(os.path.basename(image_path))[0]
    cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)

    return cmap


@click.group()
def cli():
    """A command line tool for processing images and colormaps."""
    pass


@click.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option(
    "--output",
    default="extracted_colormap.pkl",
    help="Output filename for the saved colormap pickle.",
)
def extract_colormap(image_path, output):
    """
    Extract a colormap from a PNG image by processing pixels from the center downward.
    Save the colormap as a pickle file.
    """
    cmap = create_colormap_from_image(image_path)
    with open(output, "wb") as f:
        pickle.dump(cmap, f)
    click.echo(f"Colormap extracted and saved as '{output}'.")


def load_colormap(colormap_name):
    """
    Load a colormap from a pickle file if it exists, otherwise return None.
    """
    try:
        with open(colormap_name, "rb") as f:
            cmap = pickle.load(f)
        click.echo(f"Loaded colormap from '{colormap_name}'.")
        return cmap
    except FileNotFoundError:
        return None


@click.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option(
    "--colormap",
    default=None,
    help="Name of the custom colormap to use, or the path to a pickle file.",
)
def process_image(image_path, colormap):
    """
    Process an image, map pixel RGB values to a numerical value using a colormap,
    and output a CSV file with coordinates, RGB values, and the mapped value.
    """
    # Prompt for min and max values
    min_val = click.prompt(
        "Please enter the minimum value for normalization", type=float
    )
    max_val = click.prompt(
        "Please enter the maximum value for normalization", type=float
    )

    # Try to load the colormap from a pickle file if provided
    cmap = None
    if colormap:
        cmap = load_colormap(colormap)

    # If no pickle file is found or provided, use a standard colormap or custom logic
    if cmap is None:
        if colormap:
            cmap = plt.get_cmap(colormap)
        else:
            colormap = click.prompt(
                "Please enter the name of the matplotlib colormap to use",
                default="viridis",
            )
            cmap = plt.get_cmap(colormap)

    # Load the image
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    # Define output CSV path
    output_csv = os.path.splitext(image_path)[0] + "_pixels_with_values.csv"

    with open(output_csv, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        # Write the header
        writer.writerow(["X", "Y", "R", "G", "B", "Mapped_Value"])

        # Process each pixel with a progress bar
        for y in tqdm(range(height), desc="Processing image"):
            for x in range(width):
                rgb = img.getpixel((x, y))
                mapped_value = rgb_to_value(rgb, cmap, min_val, max_val)
                writer.writerow([x, y, rgb[0], rgb[1], rgb[2], mapped_value])

    click.echo(f"CSV file with pixel values saved as '{output_csv}'.")


@click.command()
@click.option(
    "--colormap",
    default=None,
    help="Name of the custom colormap to use, or the path to a pickle file.",
)
@click.option(
    "--output", default="colormap.png", help="Output filename for the colormap PNG."
)
@click.option("--min_val", default=0, help="Minimum value for the colormap.")
@click.option("--max_val", default=100, help="Maximum value for the colormap.")
def save_colormap(colormap, output, min_val, max_val):
    """
    Generate a PNG image of the specified colormap.
    """
    # Try to load the colormap from a pickle file if provided
    cmap = None
    if colormap:
        cmap = load_colormap(colormap)

    # If no pickle file is found or provided, use a standard colormap or custom logic
    if cmap is None:
        if colormap:
            cmap = plt.get_cmap(colormap)
        else:
            colormap = click.prompt(
                "Please enter the name of the matplotlib colormap to use",
                default="viridis",
            )
            cmap = plt.get_cmap(colormap)

    # Create a figure and a colorbar
    fig, ax = plt.subplots(figsize=(6, 1), constrained_layout=True)
    fig.subplots_adjust(bottom=0.5)

    # Create a colorbar with the colormap
    norm = plt.Normalize(vmin=min_val, vmax=max_val)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation="horizontal"
    )

    # Save the colormap to a PNG file
    plt.savefig(output, dpi=300)
    click.echo(f"Colormap saved as '{output}'.")


# Add commands to the CLI group
cli.add_command(extract_colormap)
cli.add_command(process_image)
cli.add_command(save_colormap)

if __name__ == "__main__":
    cli()

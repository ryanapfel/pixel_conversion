import csv
import logging
import os
import pickle

import click
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_loggable_color_block(rgb):
    """
    Log a color block in the terminal based on the given RGB values.

    :param rgb: A tuple of (R, G, B) values, each in the range 0-255.
    """
    r, g, b = rgb
    color_block = f"\033[48;2;{r};{g};{b}m    \033[0m"

    return color_block


def get_cmap_colors(cmap):
    return [cmap(i / 255.0)[:3] for i in range(256)]


def rgb_to_value(rgb, cmap_colors, min_val=0, max_val=100):
    """
    Maps an RGB color to a numerical value using a colormap.
    """

    normalized_rgb = np.array(rgb) / 255.0

    distances = np.sqrt(np.sum((cmap_colors - normalized_rgb) ** 2, axis=1))

    idx = np.argmin(distances)

    selected_color = cmap_colors[idx]
    selected_color_rgb = tuple(int(c * 255) for c in selected_color)

    normalized_value = idx / 255.0
    value = normalized_value * (max_val - min_val) + min_val

    return value, selected_color_rgb


def create_colormap_from_image(image_path, orientation="vertical"):
    """
    Create a colormap by processing pixels from the image along a specified orientation.

    :param image_path: str
        Path to the image file.
    :param orientation: str, optional
        Orientation to process the image. Can be "vertical" (default) or "horizontal".
        - "vertical": Extract colors from the center column downwards.
        - "horizontal": Extract colors from the center row leftwards.

    :return: LinearSegmentedColormap
        A matplotlib colormap created from the extracted image colors.

    :raises ValueError:
        If the specified orientation is not "vertical" or "horizontal".
    """
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    if orientation == "vertical":
        center_x = width // 2
        # Extract colors from the center column downwards
        colors = [
            tuple(np.array(img.getpixel((center_x, y))) / 255.0) for y in range(height)
        ]
    elif orientation == "horizontal":
        center_y = height // 2
        # Extract colors from the center row leftwards
        colors = [
            tuple(np.array(img.getpixel((x, center_y))) / 255.0) for x in range(width)
        ]
    else:
        raise ValueError("Orientation must be 'vertical' or 'horizontal'.")

    # Create a colormap from the extracted colors
    cmap_name = os.path.splitext(os.path.basename(image_path))[0]
    cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)

    return cmap


def process_image_file(image_path, cmap_values, min_val, max_val):
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    image_output = []

    for y in tqdm(range(height), desc="Processing image"):
        for x in range(width):
            rgb = img.getpixel((x, y))

            mapped_value, closest_pixel_color = rgb_to_value(
                rgb, cmap_values, min_val, max_val
            )
            logger.debug(
                f"PIXEL ({x}, {y}): {get_loggable_color_block(rgb)} was most similar to {get_loggable_color_block(closest_pixel_color)}"
            )
            image_output.append(
                [image_path, x, y, rgb[0], rgb[1], rgb[2], mapped_value]
            )

    return image_output


def process_directory(directory_path, colormap, min_val, max_val, output_csv):
    """
    Process all images in a directory and save the results into a single CSV file.

    :param directory_path: str
        Path to the directory containing image files.
    :param colormap: object
        Matplotlib colormap object.
    :param min_val: int
        Minimum value for normalization.
    :param max_val: int
        Maximum value for normalization.
    :param output_csv: str
        Output CSV filename.
    """
    cmap_colors = get_cmap_colors(colormap)
    all_image_data = []

    # Process each image file in the directory
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                image_path = os.path.join(root, file)
                logger.info(f"Processing image: {image_path}")
                image_data = process_image_file(
                    image_path, cmap_colors, min_val, max_val
                )
                all_image_data.extend(image_data)

    # Write all data to the CSV file
    with open(output_csv, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Image Path", "X", "Y", "R", "G", "B", "Mapped Value"])
        csv_writer.writerows(all_image_data)

    logger.info(f"All image data has been written to '{output_csv}'.")


def load_colormap(colormap_name):
    """
    Load a colormap from a pickle file if it exists, otherwise return None.
    """
    try:
        with open(colormap_name, "rb") as f:
            cmap = pickle.load(f)
        logger.info(f"Loaded colormap from '{colormap_name}'")
        return cmap
    except FileNotFoundError:
        return None


def np_to_rgb(selected_color):
    selected_color_rgb = tuple(int(c * 255) for c in selected_color)
    return selected_color_rgb


def get_cmap_min_max_colors(cmap):
    cmap_vals = get_cmap_colors(cmap)

    min_color = get_loggable_color_block(np_to_rgb(cmap_vals[0]))
    max_color = get_loggable_color_block(np_to_rgb(cmap_vals[-1]))

    return min_color, max_color


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
@click.option(
    "--orientation", default="vertical", help="Orientation of the extracted colors."
)
def extract_colormap(image_path, output, orientation):
    """
    Extract a colormap from a PNG image by processing pixels from the center downward.
    Save the colormap as a pickle file.
    """
    cmap = create_colormap_from_image(image_path, orientation)
    with open(output, "wb") as f:
        pickle.dump(cmap, f)
    click.echo(f"Colormap extracted and saved as '{output}'.")


@click.command()
@click.argument("directory_path", type=click.Path(exists=True))
@click.option(
    "--colormap",
    default=None,
    help="Name of the custom colormap to use, or the path to a pickle file.",
)
@click.option(
    "--output_csv",
    default="output.csv",
    help="Name of the output CSV file containing the aggregated results.",
)
def process_dir(directory_path, colormap, output_csv):
    """
    Process all images in a directory, map pixel RGB values to numerical values using a colormap,
    and save the results to a single CSV file.

    :param directory_path: str
        Path to the directory containing image files.
    :param colormap: str
        Name or path to the colormap.
    :param output_csv: str
        Name of the output CSV file.
    """
    # Load colormap
    cmap = load_colormap(colormap)
    if not cmap:
        cmap = plt.get_cmap("turbo")
        logger.warning("Colormap not provided, using default 'turbo' colormap.")

    min_color, max_color = get_cmap_min_max_colors(cmap)

    min_val = click.prompt(
        f"Please enter the minimum value for normalization, associated with {min_color}",
        type=float,
        default=0,
    )
    max_val = click.prompt(
        "Please enter the maximum value for normalization, associated with "
        f"{max_color}",
        type=float,
        default=100,
    )

    # Process the directory
    process_directory(directory_path, cmap, min_val, max_val, output_csv)
    click.echo(f"Processing completed. Results saved in '{output_csv}'.")


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
cli.add_command(process_dir)
cli.add_command(save_colormap)

if __name__ == "__main__":
    cli()

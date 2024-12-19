import logging
import os

import matplotlib.pyplot as plt
import pytest

from src.script import (
    create_colormap_from_image,
    get_cmap_colors,
    get_cmap_min_max_colors,
    get_loggable_color_block,
    load_colormap,
    process_image_file,
    rgb_to_value,
    save_colormap,
)

logger = logging.getLogger(__name__)


@pytest.fixture()
def cmap():
    return plt.get_cmap("turbo")


@pytest.fixture()
def cmap_colors_np(cmap):
    return get_cmap_colors(cmap)


@pytest.fixture()
def image_path():
    return "tests/test_images/test_image.png"


def test_load_colormap():
    # Test loading a valid colormap pickle file
    valid_file = "colormaps/custom_colormap.pkl"
    assert load_colormap(valid_file) is not None


def test_create_colormap_from_image():
    # Test creating a colormap from an image
    image_path = "tests/test_images/test_image_2.png"
    cmap = create_colormap_from_image(image_path)

    assert cmap is not None


def test_loggable_color_block():
    block = get_loggable_color_block((255, 0, 0))
    logger.info(f'color block, should be red: "{block}"')

    block = get_loggable_color_block((0, 255, 0))
    logger.info(f'color block, should be green: "{block}"')
    block = get_loggable_color_block((0, 0, 255))
    logger.info(f'color block, should be blue: "{block}"')
    # visual testing here, as long no error thrown

    assert True


def test_create_colormap_from_sample():
    """
    Test creating a colormap from a sample image file and log color blocks for visual inspection.
    """
    # Sample colormap file path
    sample_image_path = "colormaps/colormap_vertical.png"

    # Ensure the file exists
    assert os.path.exists(
        sample_image_path
    ), f"Sample file {sample_image_path} does not exist."

    # Create a colormap
    cmap = create_colormap_from_image(sample_image_path, orientation="vertical")

    # Verify the colormap object
    assert cmap is not None, "Colormap creation failed."

    # Log a few color blocks for visual inspection
    logger.info("Logging first few color blocks from the colormap:")
    for i in range(0, 256, 20):  # Sample a few points (e.g., every 50th step)
        color = tuple(
            int(c * 255) for c in cmap(i / 255.0)[:3]
        )  # Get RGB from colormap
        color_block = get_loggable_color_block(color)
        logger.info(f"Color {i}: {color} {color_block}")

    # No errors should occur during this process
    assert True


def test_rgb_to_value(cmap_colors_np):
    # Test mapping an RGB color to a numerical value
    rgb = (255, 0, 0)
    min_val = 0
    max_val = 100
    red_value, selected_color = rgb_to_value(
        rgb, cmap_colors=cmap_colors_np, min_val=min_val, max_val=max_val
    )

    _color = get_loggable_color_block(selected_color)
    _source_color = get_loggable_color_block(rgb)

    logger.info(f"SOURCE RGB: {rgb}  COLOR: {_source_color}")
    logger.info(f"VALUE: {red_value}  COLOR: {_color}")
    assert 50 <= red_value <= 100


def test_process_image(image_path, cmap):
    # Test processing an image and mapping RGB values to numerical values
    min_val = 0
    max_val = 100
    process_image_file(image_path, colormap=cmap, min_val=min_val, max_val=max_val)


def test_min_max_values(cmap):
    # Test processing an image and mapping RGB values to numerical values

    min_color, max_color = get_cmap_min_max_colors(cmap)

    logger.info(f"MIN COLOR: {min_color}")
    logger.info(f"MAX COLOR: {max_color}")

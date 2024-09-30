## Usage

1. Extract Color Map

```
poetry run python script_name.py extract-colormap screenshot.png --output='custom_colormap.pkl
```

2. Processing an Image with the Extracted Colormap:

```
poetry run python script_name.py process-image input_image.png --colormap='custom_colormap.pkl'
```

3. Saving the Colormap as a PNG:

```
poetry run python script_name.py save-colormap --colormap='custom_colormap.pkl' --output='custom_colormap.png'
```

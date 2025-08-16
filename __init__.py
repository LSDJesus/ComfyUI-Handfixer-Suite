# __init__.py

from .nodes import MediaPipeDetailer

# A dictionary that maps the node's class name to the node's class
NODE_CLASS_MAPPINGS = {
    "MediaPipeDetailer": MediaPipeDetailer
}

# A dictionary that contains the friendly display names of the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MediaPipeDetailer": "MediaPipe Detailer Suite"
}

# A manifest file for the ComfyUI Manager
MANIFEST = {
    "name": "ComfyUI HandFixer Suite",
    "version": (1,0,0),
    "author": "Your GitHub Name",
    "project": "https://github.com/YOUR_USERNAME/comfyui-handfixer-suite",
    "description": "An all-in-one ADetailer-style inpainting suite powered by MediaPipe for fixing hands, faces, and more.",
}


print("### ComfyUI-Handfixer-Suite: Loaded a new and improved version.")
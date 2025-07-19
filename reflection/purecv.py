from PIL import Image, ImageOps, ImageFilter
import numpy as np
from rembg import remove
import time


def generate_reflection_on_original_bg_full_height(
    image_path,
    output_path,
    reflection_strength=0.4,
    blur_radius=1.0,
    reflection_y_offset=0,
    fade_power=1.5,
):
    """
    Generates a reflection effect for an object in an image,
    extending the reflection to the bottom of the image,
    keeping the original background.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the output image.
        reflection_strength (float): Max opacity of the reflection's top (0.0 to 1.0).
        blur_radius (float): Gaussian blur radius for the reflection.
        reflection_y_offset (int): Vertical gap between object and reflection.
        fade_power (float): Controls the fade-off curve. >1 fades faster initially.
    """
    try:
        original_img_pil = Image.open(image_path).convert("RGBA")
        img_width, img_height = original_img_pil.size
    except FileNotFoundError:
        print(f"Error: Input image '{image_path}' not found.")
        return
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    try:
        print("Performing background removal with rembg...")
        object_with_alpha = remove(original_img_pil)
        print("Background removal complete.")
    except Exception as e:
        print(f"Error during object segmentation with rembg: {e}")
        return

    bbox = object_with_alpha.getbbox()
    if not bbox:
        print("Could not detect object bounds after segmentation.")
        return

    print(f"Object bounding box: {bbox}")
    segmented_object_crop = object_with_alpha.crop(bbox)
    obj_width, obj_height = segmented_object_crop.size
    print(f"Object dimensions: {obj_width}x{obj_height}")

    if obj_width == 0 or obj_height == 0:
        print("Detected object has zero width or height.")
        return

    object_pixels_for_reflection = original_img_pil.crop(bbox)
    reflection_base = ImageOps.flip(object_pixels_for_reflection)
    object_alpha_mask = segmented_object_crop.split()[-1]
    reflection_alpha_mask_flipped = ImageOps.flip(object_alpha_mask)

    print("Creating gradient fade effect...")
    np_alpha_mask = np.array(reflection_alpha_mask_flipped, dtype=np.float32)
    y_coords = np.arange(obj_height)
    gradient_values = (
        (1 - (y_coords / obj_height)) ** fade_power * reflection_strength * 255
    )
    print(
        f"Gradient values shape: {gradient_values.shape}, min: {gradient_values.min()}, max: {gradient_values.max()}"
    )
    gradient = gradient_values.reshape(obj_height, 1)
    alpha_fade_gradient = np.tile(gradient, (1, obj_width))
    print(f"Alpha fade gradient shape: {alpha_fade_gradient.shape}")
    faded_reflection_alpha_np = np.minimum(np_alpha_mask, alpha_fade_gradient).astype(
        np.uint8
    )
    print(
        f"Faded reflection alpha min: {faded_reflection_alpha_np.min()}, max: {faded_reflection_alpha_np.max()}"
    )
    faded_reflection_alpha_pil = Image.fromarray(faded_reflection_alpha_np, mode="L")

    reflection_image_final = reflection_base.copy()
    reflection_image_final.putalpha(faded_reflection_alpha_pil)

    if blur_radius > 0:
        reflection_image_final = reflection_image_final.filter(
            ImageFilter.GaussianBlur(radius=blur_radius)
        )

    reflection_start_y_on_canvas = bbox[3] + reflection_y_offset
    max_reflection_height_on_canvas = img_height - reflection_start_y_on_canvas
    print(f"Reflection start Y: {reflection_start_y_on_canvas}")
    print(f"Max reflection height available: {max_reflection_height_on_canvas}")

    if max_reflection_height_on_canvas <= 0:
        print("Object is too low, or offset is too large; no space for reflection.")
        original_img_pil.save(output_path)
        return

    # The reflection to paste will be cropped from the source reflection.
    # Its height will be the minimum of its own full height (obj_height)
    # and the available space on the canvas.
    height_to_crop_from_source_reflection = min(
        obj_height, max_reflection_height_on_canvas
    )
    print(f"Final reflection height to use: {height_to_crop_from_source_reflection}")

    if height_to_crop_from_source_reflection <= 0:
        print("Calculated reflection height is zero or less. Nothing to paste.")
        original_img_pil.save(output_path)
        return

    reflection_to_paste = reflection_image_final.crop(
        (0, 0, obj_width, height_to_crop_from_source_reflection)
    )
    print(f"Cropped reflection size: {reflection_to_paste.size}")

    # Composite the reflection onto the original image
    output_image = original_img_pil.copy()
    paste_x = bbox[0]
    print(
        f"Pasting reflection at position: x={paste_x}, y={reflection_start_y_on_canvas}"
    )
    output_image.paste(
        reflection_to_paste,
        (paste_x, reflection_start_y_on_canvas),
        reflection_to_paste,
    )

    try:
        if original_img_pil.info.get(
            "format", ""
        ).upper() == "JPEG" or image_path.lower().endswith((".jpg", ".jpeg")):
            output_image = output_image.convert("RGB")
        output_image.save(output_path)
        print(f"Successfully generated full-height reflection: {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")


if __name__ == "__main__":
    start_time = time.time()

    input_image = "test.webp"
    output_image_full_reflection = "test_output.png"
    generate_reflection_on_original_bg_full_height(
        image_path=input_image,
        output_path=output_image_full_reflection,
        reflection_strength=0.25,
        blur_radius=0.8,
        reflection_y_offset=1,
        fade_power=2,
    )

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Processing complete. Check '{output_image_full_reflection}'.")
    print(f"Total execution time: {execution_time:.2f} seconds")

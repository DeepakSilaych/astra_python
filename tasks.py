import os
import tempfile
import base64

import reflection.purecv as purecv
from celery_worker import celery_app


@celery_app.task(bind=True)
def reflect_task(
    self,
    file_bytes,
    suffix,
    reflection_strength,
    blur_radius,
    reflection_y_offset,
    fade_power,
):
    """
    Celery task to generate a reflection using purecv.
    Returns a base64-encoded PNG image.
    """
    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, f"input{suffix}")
        output_path = os.path.join(tmpdir, f"output{suffix}")
        # Write uploaded bytes to disk
        with open(input_path, "wb") as f:
            f.write(file_bytes)
        # Run the reflection generator
        purecv.generate_reflection_on_original_bg_full_height(
            image_path=input_path,
            output_path=output_path,
            reflection_strength=reflection_strength,
            blur_radius=blur_radius,
            reflection_y_offset=reflection_y_offset,
            fade_power=fade_power,
        )
        # Read the generated image
        with open(output_path, "rb") as f:
            output_bytes = f.read()
    # Return base64-encoded result for JSON compatibility
    return base64.b64encode(output_bytes).decode("ascii")

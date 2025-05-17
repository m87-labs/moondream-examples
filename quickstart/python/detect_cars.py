import os
import json
from PIL import Image, ImageDraw
import moondream as md

# This example detects cars in all images in a folder.
# It saves overlay images with bounding boxes and a JSON summary.

# Initialize Moondream for local inference
model = md.vl(endpoint="http://localhost:2020/v1")
# For Moondream Cloud, use your API key:
# model = md.vl(api_key="<your-api-key>")

INPUT_FOLDER = os.path.join(os.path.dirname(__file__), "..", "..", "images")
OUTPUT_FOLDER = os.path.join(INPUT_FOLDER, "detected")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

results = []

for file_name in os.listdir(INPUT_FOLDER):
    if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    input_path = os.path.join(INPUT_FOLDER, file_name)
    image = Image.open(input_path)

    # Detect cars in the image
    detection = model.detect(image, "car")["objects"]

    # Draw bounding boxes on a copy of the image
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for box in detection:
        draw.rectangle(
            [box["x_min"], box["y_min"], box["x_max"], box["y_max"]],
            outline="red",
            width=3,
        )

    output_path = os.path.join(OUTPUT_FOLDER, file_name)
    overlay.save(output_path)

    results.append({
        "original": input_path,
        "overlay": output_path,
        "boxes": detection,
    })

# Save detection results to JSON
json_path = os.path.join(OUTPUT_FOLDER, "results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved {len(results)} results to {json_path}")

import os
import json
from PIL import Image, ImageDraw
import moondream as md

# This example shows off every vision capability on a folder of images.
# For each image we generate a caption, answer a question, detect cars,
# and point to the center of each car. The results are written to JSON
# and overlay images with bounding boxes and points are saved.

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

    # Run all capabilities on the image
    caption = model.caption(image)["caption"]
    answer = model.query(image, "What's in this image?")["answer"]
    detection = model.detect(image, "car")["objects"]
    points = model.point(image, "car")["points"]

    # Draw bounding boxes on a copy of the image
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for box in detection:
        draw.rectangle(
            [box["x_min"], box["y_min"], box["x_max"], box["y_max"]],
            outline="red",
            width=3,
        )
    for point in points:
        r = 4
        draw.ellipse(
            [point["x"] - r, point["y"] - r, point["x"] + r, point["y"] + r],
            fill="blue",
        )

    output_path = os.path.join(OUTPUT_FOLDER, file_name)
    overlay.save(output_path)

    results.append({
        "original": input_path,
        "overlay": output_path,
        "caption": caption,
        "answer": answer,
        "boxes": detection,
        "points": points,
    })

# Save detection results to JSON
json_path = os.path.join(OUTPUT_FOLDER, "results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved {len(results)} results to {json_path}")

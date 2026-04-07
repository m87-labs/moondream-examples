"""Simple detect finetuning example for iShape antenna detection.

Dataset: moondream/ishape-antenna

Requires:
    pip install datasets pillow pycocotools

To run with a local moondream-python checkout:
    PYTHONPATH=/path/to/moondream-python python train_ishape_antenna_detect.py

Set MOONDREAM_API_KEY.
Optional: HF_TOKEN, MOONDREAM_TUNING_ENDPOINT.
"""

import io
import os
import time

from datasets import load_dataset
from PIL import Image
from pycocotools import mask as mask_utils

import moondream as md

DATASET_NAME = "moondream/ishape-antenna"
OBJECT_NAME = "antenna"

STEPS = 100
BATCH_SIZE = 32
NUM_ROLLOUTS = 8
EVAL_EVERY = 10
EVAL_LIMIT = 50
LR = 2e-4
RANK = 32
SEED = 42
MAX_TOKENS = 256
MAX_OBJECTS = None

TUNING_ENDPOINT = os.environ.get(
    "MOONDREAM_TUNING_ENDPOINT",
    "https://api.moondream.ai/v1/tuning",
)


def decode_image(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB") if image.mode != "RGB" else image
    if image.get("bytes") is not None:
        with Image.open(io.BytesIO(image["bytes"])) as decoded:
            return decoded.convert("RGB")
    with Image.open(image["path"]) as decoded:
        return decoded.convert("RGB")


def rle_to_box(rle):
    height, width, counts = rle.split(" ", 2)
    x, y, w, h = mask_utils.toBbox({
        "size": [int(height), int(width)],
        "counts": counts.encode("ascii"),
    }).tolist()
    return {
        "x_min": x / int(width),
        "y_min": y / int(height),
        "x_max": (x + w) / int(width),
        "y_max": (y + h) / int(height),
    }


def load_examples(split, limit=None):
    rows = load_dataset(
        DATASET_NAME,
        split=split,
        token=os.environ.get("HF_TOKEN"),
    ).shuffle(seed=SEED)

    examples = []
    for row in rows:
        boxes = [rle_to_box(obj["rle"]) for obj in row["objects"] if obj["name"] == OBJECT_NAME]
        if not boxes:
            continue
        examples.append({"image": decode_image(row["image"]), "boxes": boxes})
        if limit is not None and len(examples) == limit:
            break
    return examples


def detect_settings(temperature):
    settings = {"temperature": temperature, "max_tokens": MAX_TOKENS}
    if MAX_OBJECTS is not None:
        settings["max_objects"] = MAX_OBJECTS
    return settings


def request_stream(examples):
    while True:
        for example in examples:
            yield example, {
                "skill": "detect",
                "image": example["image"],
                "object": OBJECT_NAME,
                "num_rollouts": NUM_ROLLOUTS,
                "settings": detect_settings(1.0),
                "ground_truth": {"boxes": example["boxes"]},
            }


def evaluate(ft, examples):
    miou = 0.0
    for example in examples:
        response = ft.rollouts(
            "detect",
            image=example["image"],
            object=OBJECT_NAME,
            settings=detect_settings(0.0),
            ground_truth={"boxes": example["boxes"]},
        )
        miou += response["rewards"][0]
    return miou / len(examples)


def main():
    train_examples = load_examples("train")
    eval_examples = load_examples("validation", limit=EVAL_LIMIT)

    ft = md.ft(
        api_key=os.environ["MOONDREAM_API_KEY"],
        endpoint=TUNING_ENDPOINT,
        name=f"ishape-antenna-detect-{int(time.time())}",
        rank=RANK,
    )
    print(f"Created finetune: {ft.finetune_id} ({ft.name})", flush=True)

    batch = []
    for _, response in ft.rollout_stream(request_stream(train_examples)):
        batch.append({
            "mode": "rl",
            "request": response["request"],
            "rollouts": response["rollouts"],
            "rewards": response["rewards"],
        })
        if len(batch) < BATCH_SIZE:
            continue

        step = ft.train_step(batch, lr=LR)
        batch = []
        print(f"step={step['step']} reward_mean={step['reward_mean']:.4f}", flush=True)

        if step["step"] % EVAL_EVERY == 0 or step["step"] == STEPS:
            eval_miou = evaluate(ft, eval_examples)
            ft.log_metrics(step=step["step"], metrics={"eval/miou": eval_miou})
            print(f"eval step={step['step']} miou={eval_miou:.4f}", flush=True)

        if step["step"] == STEPS:
            break

    checkpoint = ft.save_checkpoint()["checkpoint"]
    print(f"Saved checkpoint: {checkpoint['checkpoint_id']}", flush=True)
    print(f"Model ID: {ft.model(checkpoint['step'])}", flush=True)


if __name__ == "__main__":
    main()

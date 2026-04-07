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
import math
import os
import statistics
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
GROUND_TRUTH_INJECTION_STD_THRESH = 0.1
GROUND_TRUTH_INJECTION_MAX_REWARD = 0.4
GROUND_TRUTH_INJECTION_REWARD_SCALE = 2.0

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


def quantize_coord(value):
    return max(0, min(1023, int(round(float(value) * 1023))))


def quantize_size(value):
    value = max(float(value), 2**-10)
    return max(0, min(1023, int(round((math.log2(value) + 10) / 10 * 1023))))


def pad_boxes(boxes, target_boxes):
    padded = list(boxes[:target_boxes])
    while len(padded) < target_boxes:
        padded.append({"x_min": 0.0, "y_min": 0.0, "x_max": 0.0, "y_max": 0.0})
    return padded


def encode_boxes(boxes, target_boxes):
    coords = []
    sizes = []
    for box in boxes[:target_boxes]:
        x_center = (box["x_min"] + box["x_max"]) / 2.0
        y_center = (box["y_min"] + box["y_max"]) / 2.0
        width = box["x_max"] - box["x_min"]
        height = box["y_max"] - box["y_min"]
        coords.extend([quantize_coord(x_center), quantize_coord(y_center)])
        sizes.append([quantize_size(width), quantize_size(height)])

    while len(sizes) < target_boxes:
        coords.extend([0, 0])
        sizes.append([0, 0])

    return coords, sizes


def rollout_box_count(rollout):
    coords = rollout.get("coords", [])
    sizes = rollout.get("sizes", [])
    if len(coords) % 2 != 0:
        return None
    coord_boxes = len(coords) // 2
    if sizes and coord_boxes and len(sizes) != coord_boxes:
        return None
    if sizes:
        return len(sizes)
    if coord_boxes:
        return coord_boxes
    return len(rollout.get("output", {}).get("objects", []))


def detect_token_ids(rollout):
    tokens = rollout.get("answer_tokens", [])
    if len(tokens) < 3:
        return None
    return tokens[0], tokens[2]


def build_ground_truth_rollout(rollout, gt_boxes, target_boxes):
    token_ids = detect_token_ids(rollout)
    if token_ids is None:
        return None

    coord_id, size_id = token_ids
    injected_boxes = pad_boxes(gt_boxes, target_boxes)
    coords, sizes = encode_boxes(injected_boxes, target_boxes)
    return {
        "skill": rollout.get("skill", "detect"),
        "finish_reason": rollout["finish_reason"],
        "output": {"objects": injected_boxes},
        "answer_tokens": [coord_id, coord_id, size_id] * target_boxes,
        "thinking_tokens": list(rollout.get("thinking_tokens", [])),
        "has_answer_separator": rollout.get("has_answer_separator", False),
        "coords": coords,
        "sizes": sizes,
    }


def maybe_inject_ground_truth(rollouts, rewards, gt_boxes):
    if not rollouts or not rewards:
        return rollouts, rewards, 0

    reward_std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    max_reward = max(rewards)
    if (
        reward_std >= GROUND_TRUTH_INJECTION_STD_THRESH
        or max_reward >= GROUND_TRUTH_INJECTION_MAX_REWARD
    ):
        return rollouts, rewards, 0

    replace_idx = min(range(len(rewards)), key=rewards.__getitem__)
    target_boxes = rollout_box_count(rollouts[replace_idx]) or len(gt_boxes)
    if target_boxes <= 0:
        return rollouts, rewards, 0

    injected_rollout = build_ground_truth_rollout(
        rollouts[replace_idx],
        gt_boxes,
        target_boxes,
    )
    if injected_rollout is None:
        return rollouts, rewards, 0

    rollouts = list(rollouts)
    rewards = list(rewards)
    rollouts[replace_idx] = injected_rollout
    rewards[replace_idx] = max(
        0.5,
        min(1.0, GROUND_TRUTH_INJECTION_REWARD_SCALE * max_reward),
    )
    return rollouts, rewards, 1


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
    batch_gt_injected = 0
    for example, response in ft.rollout_stream(request_stream(train_examples)):
        rollouts, rewards, gt_injected = maybe_inject_ground_truth(
            response["rollouts"],
            response["rewards"],
            example["boxes"],
        )
        batch_gt_injected += gt_injected
        batch.append({
            "mode": "rl",
            "request": response["request"],
            "rollouts": rollouts,
            "rewards": rewards,
        })
        if len(batch) < BATCH_SIZE:
            continue

        step = ft.train_step(batch, lr=LR)
        batch = []
        print(
            f"step={step['step']} reward_mean={step['reward_mean']:.4f} gt_injected={batch_gt_injected}",
            flush=True,
        )
        batch_gt_injected = 0

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

# Moondream Python Client Library

Official Python client library for Moondream, a fast multi-function VLM. This client can target either the [Moondream Cloud](https://moondream.ai/cloud) or a [Moondream Server](https://moondream.ai/server). Both are free, though the cloud has a limits on the free tier.

## Capabilities
Moondream goes beyond the typical VLM "query" ability to include more visual functions. These include:

- **caption**: Generate descriptive captions for images
- **query**: Ask questions about image content
- **detect**: Find bounding boxes around objects in images
- **point**: Identify the center location of specified objects in images

You can try this out anytime on [Moondream's playground](https://moondream.ai/playground).

## Installation

Install the package from PyPI:

```bash
pip install moondream
```

## Quick Start

Choose how you want to run it:

1. **Moondream Cloud**: (with 5,000 free requests/day): get a free API key from [the Moondream cloud console](https://moondream.ai/c/cloud/api-keys).
2. **Moondream Server**: Run it locally by installing and running [the Moondream server](https://mooondream.ai/moondream-server).

Once you've done at least *one* of these, try running this code:

```python
import moondream as md
from PIL import Image

# Initialize for Moondream Cloud
model = md.vl(api_key="<your-api-key>")

# ...or initialize for a local Moondream Server
model = md.vl(endpoint="http://localhost:2020/v1")

# Load an image
image = Image.open("path/to/image.jpg")

# Generate a caption
caption = model.caption(image)["caption"]
print("Caption:", caption)

# Ask a question
answer = model.query(image, "What's in this image?")["answer"]
print("Answer:", answer)

# Stream the response
for chunk in model.caption(image, stream=True)["caption"]:
    print(chunk, end="", flush=True)
```

## API Reference

### Constructor

```python
# Cloud inference
model = md.vl(api_key="<your-api-key>")

# Local inference
model = md.vl(endpoint="http://localhost:2020/v1")
```

### Methods

#### caption(self, image: Union[Image.Image, EncodedImage], length: Literal["normal", "short", "long"] = "normal", stream: bool = False) -> CaptionOutput

Generate a caption for an image.

```python
caption = model.caption(image, length="short")["caption"]
print(caption)

# Generate a caption with streaming (default: False)
for chunk in model.caption(image, length="short", stream=True)["caption"]:
    print(chunk, end="", flush=True)
```

#### query(self, image: Union[Image.Image, EncodedImage], question: str, stream: bool = False) -> QueryOutput

Ask a question about an image.

```python
answer = model.query(image, question="What's in this image?")["answer"]
print("Answer:", answer)

# Ask a question with streaming (default: False)
for chunk in model.query(image, question="What's in this image?", stream=True)["answer"]:
    print(chunk, end="", flush=True)
```

#### detect(self, image: Union[Image.Image, EncodedImage], object: str) -> DetectOutput

Detect specific objects in an image.

```python
detect_output = model.detect(image, "item")["objects"]
print(detect_output)
```

#### point(self, image: Union[Image.Image, EncodedImage], object: str) -> PointOutput

Get coordinates of specific objects in an image.

```python
point_output = model.point(image, "person")
print(point_output)
```

#### encode_image(self, image: Union[Image.Image, EncodedImage]) -> Base64EncodedImage

Produce Base64EncodedImage.

```python
encoded_image = model.encode_image(image)
```

### Image Types

- Image.Image: PIL Image object
- Base64EncodedImage: Object produced by model.encode_image(image), subtype of EncodedImage

### Response Types

- CaptionOutput: `{"caption": str | Generator}`
- QueryOutput: `{"answer": str | Generator}`
- DetectOutput: `{"objects": List[Region]}`
- PointOutput: `{"points": List[Point]}`
- Region: Bounding box with coordinates (`x_min`, `y_min`, `x_max`, `y_max`)
- Point: Coordinates (`x`, `y`) indicating the object center

## Links

- [Website](https://moondream.ai/)
- [Try it out on the free playground](https://moondream.ai/playground)
- [GitHub](https://github.com/vikhyat/moondream)
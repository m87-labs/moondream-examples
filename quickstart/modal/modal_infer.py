import moondream as md
from PIL import Image

# This is where your copied endpoint URL will be utilized.
# Make sure to add /v1 at the end of the URL like shown below.
# Your endpoint URL may look different, depending on your setup.
# Replace "username" with your actual username on modal labs.
model = md.vl(endpoint="https://yourname--moondream-server.modal.run/v1")

image = Image.open("../../images/frieren.jpg")

# Query
answer = model.query(image, "What's in this image?")["answer"]
print(answer)

# Streaming captions
response = model.caption(image, "What's in this image?", stream=True)
for chunk in response["answer"]:
    print(chunk, end="", flush=True)

# You can use your own object name in the object parameter.
answer = model.detect(image, object="burger")
print(answer)

# Similar to detect, you can use your own object name in object parameter.
answer = model.point(image, object="object-name")
print(answer)

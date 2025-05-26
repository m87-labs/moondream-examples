import moondream as md
from PIL import Image

# this is where your copied endpoint URL will be utilized.
# make sure to add /v1 at the end of the URL like shown below.
# your endpoint URL may look different, depending on your setup
# replace "username" with your actual username on modal labs
model = md.vl(endpoint="https://ethan-4--moondream-server-dev.modal.run")

image = Image.open("../../images/frieren.jpg")

# query
answer = model.query(image, "What's in this image?")["answer"]
print(answer)

# streaming captions
response = model.caption(image, "What's in this image?", stream=True)
for chunk in response["answer"]:
    print(chunk, end="", flush=True)

# or you can use your own object name in the object parameter
answer = model.detect(image, object="burger")
print(answer)

# similar to detect, you can use your own object name in object parameter
answer = model.point(image, object="object-name")
print(answer)

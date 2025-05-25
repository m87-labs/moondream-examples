import moondream as md
from PIL import Image

# your endpoint URL may look different, depending on your setup
# make sure to add /v1 at the end of the URL
model = md.vl(endpoint="https://username--moondream-server.modal.run/v1")

# this will take a local image file and send it to moondream station on modal labs
image = Image.open("../images/frieren.jpg")

# query
answer = model.query(image, "What's in this image?")["answer"]

# other functions:
# answer = model.detect(image, object="cat")
# answer = model.point(image, "cat")
print(answer)

# caption with streaming
response = model.caption(image, length="normal", stream=True)
for chunk in response["caption"]:
    print(chunk, end="", flush=True)

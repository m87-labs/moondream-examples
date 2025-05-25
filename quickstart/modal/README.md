# Deploy Moondream Station on Modal Labs

Deploy Moondream vision AI as a web service with GPU acceleration.

## Setup

### **Create Modal account**
Go to [modal.com](https://modal.com) to sign up.

### **Install Modal Labs Python Library**:
To access Modal Labs deployment tools, you need to install the library:

   ```bash
   pip install modal
   ```

### **Login**
   To connect your computer to your modal labs account, run:
   ```bash
   modal token new
   ```
   This will open an authorization page in your web browser where you can authenticate your system with a login token.

### **Creating a Moondream Station Instance**

#### Setup for Deployment

We can now create a program to host Moondream Station on Modal Labs, which will allow us to access Moondream from anywhere across the world. This is done by forwarding the port used by Moondream Station in the Modal Labs container using a webserver.

   ```python
   # moondream_station.py
   import modal
   import subprocess
   import time

# you can use your own app name
   app = modal.App("moondream")

   image = modal.Image.from_registry("ubuntu:22.04", add_python="3.11").run_commands([
       "apt-get update && apt-get install -y curl",
       "curl -fsSL https://depot.moondream.ai/station/install.sh | bash",
   ])

   @app.function(image=image, memory=4096, gpu="L4", timeout=86400, min_containers=1, max_containers=1)
   @modal.web_server(2020, startup_timeout=300.0)
   def server():
       subprocess.Popen(["/moondream_station"])

   @app.local_entrypoint()
   def main():
       try:
           while True:
               time.sleep(60)
       except KeyboardInterrupt:
           pass
   ```

Note: For more functionalities like scaling, whitelisting, TCP tunnels, FastAPI support, check out [Modal Labs' documentation](https://modal.com/docs).

#### **Deploy**:

To deploy, simply execute the following:
   ```bash
   modal deploy moondream_station.py
   ```
This launches Moondream Station on Modal Labs. 

To stop your instance, run:

```
modal app stop moondream
```
You may also locate and stop the app from your Modal Labs dashboard.

### **Get your URL**
Modal will show your server URL at the start of execution. It may look something like:
   ```
   https://yourname--moondream-server-dev.modal.run
   ```
   This is your permanent service endpoint.


## Inference

To use moondream station, you can access the modal labs endpoint using our Python client.

For example:
```python
# main.py
import moondream as md
from PIL import Image

# to use with the example code, make sure to add /v1 at the end
model = md.vl(endpoint="https://yourname--moondream-server-dev.modal.run/v1")
image = Image.open("/path/to/image")

# query
answer = model.query(image, "What's in this image?")["answer"]
print(answer)

# streaming captions
response = model.caption(image, "What's in this image?", stream=True)
for chunk in response["answer"]:
    print(chunk, end="", flush=True)

answer = model.detect(image, object="object-name")
print(answer)

answer = model.point(image, "object-name")
print(answer)
```
import modal
import subprocess

app = modal.App("moondream")

# this will create a container image on Modal Labs and install Moondream Station
image = modal.Image.from_registry("ubuntu:22.04", add_python="3.11").run_commands(
    [
        "apt-get update && apt-get install -y curl",
        "curl -fsSL https://depot.moondream.ai/station/install.sh | bash",
    ]
)


@app.function(
    image=image,
    memory=4096,
    gpu="L4",
    timeout=86400,
    min_containers=1,
    max_containers=1,
)
@modal.web_server(2020, startup_timeout=300.0)
def server():
    subprocess.Popen(["/moondream_station"])

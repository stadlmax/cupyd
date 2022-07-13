import modules.conda
import modules.cuda as cuda
import modules.jupyter


def gnn_env(writer):
    writer.emit("RUN apt-get update && apt-get install -y --no-install-recommends graphviz")
    writer.emit("COPY contexts/envs/gnn_dev.yaml /tmp/gnn_dev.yaml")
    writer.condaEnv("/tmp/gnn_dev.yaml", "gnn_dev", deleteYaml=True)
    
    writer.emit('SHELL ["conda", "run", "-n", "gnn_dev", "/bin/bash", "-c"]')
    writer.emit("RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu115.html")

def emit(writer, **kwargs):
    modules.conda.emit(writer)
    if "cudaVersion" not in kwargs:
        raise Exception("'cudaVersion' is mandatory!")
    _, _, cudaVersionShort, _ = cuda.shortVersion(kwargs["cudaVersion"])
    gnn_env(writer)
    modules.jupyter.emit(writer, **kwargs)
    writer.emit("COPY contexts/envs/gnn-dev /gnn-dev")

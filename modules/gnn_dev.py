import modules.conda
import modules.cuda as cuda
import modules.jupyter


def gnn_env(writer):
    writer.emit("COPY contexts/envs/gnn_dev.yaml /tmp/gnn_dev.yaml")
    writer.emit("""RUN \\
    mamba env update -n base -f /tmp/gnn_dev.yaml && \\
    rm -f /tmp/gnn_dev.yaml && \\
    mamba clean --yes --all""")
    writer.emit("RUN python3 -m pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html")
    writer.emit("RUN python3 -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html")


def emit(writer, **kwargs):
    modules.conda.emit(writer)
    if "cudaVersion" not in kwargs:
        raise Exception("'cudaVersion' is mandatory!")
    _, _, cudaVersionShort, _ = cuda.shortVersion(kwargs["cudaVersion"])
    gnn_env(writer)
    modules.jupyter.emit(writer, **kwargs)
    writer.emit("COPY contexts/envs/gnn-dev /gnn-dev")

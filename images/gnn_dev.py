from __future__ import absolute_import
import modules.dev_env
import modules.gnn_dev
import modules.cuda
import modules.cuda_dev
import modules.internal


def emit(writer, **kwargs):
    if "cudaVersion" not in kwargs:
        raise Exception("'cudaVersion' is mandatory!")
    if "base" not in kwargs:
        raise Exception("'base' is mandatory!")
    if "rcUrl" not in kwargs:
        kwargs["rcUrl"] = None
    modules.cuda_dev.emit(writer, kwargs["cudaVersion"], kwargs["base"],
                          kwargs["rcUrl"])
    modules.gnn_dev.emit(writer, **kwargs)
    modules.dev_env.emit(writer, **kwargs)


def images():
    imgs = {}
    for osVer in ["20.04"]:
        verStr = osVer.replace(".", "")
        for cudaVer in ["11.5"]:
            _, _, cu_short, _ = modules.cuda.shortVersion(cudaVer)
            cu_short = cu_short.replace(".", "")
            imgName = "gnn-dev:%s-%s" % (verStr, cu_short)
            imgs[imgName] = {
                "cudaVersion": cudaVer,
                "base": "ubuntu:%s" % osVer,
                "needsContext": True,
            }
    imgs.update(modules.internal.read_rc())
    return imgs

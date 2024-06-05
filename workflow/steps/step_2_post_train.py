# step_2_post_train.py
# Created by Yichao

# region
import os



from dpgen.generator.lib.utils import (
    make_iter_name,
    log_task
)


from config.config import *


from dpgen.util import convert_training_data_to_hdf5, expand_sys_str
# endregion


def post_train(iter_index, jdata, mdata, base_dir):
    # load json param
    numb_models = jdata["numb_models"]
    # paths
    iter_name = make_iter_name(iter_index)
    # work_path = os.path.join(iter_name, train_name)
    work_path = os.path.join(base_dir, iter_name, train_name)
    # check if is copied
    copy_flag = os.path.join(work_path, "copied")
    if os.path.isfile(copy_flag):
        log_task("copied model, do not post train")
        return
    # symlink models
    for ii in range(numb_models):
        if not jdata.get("dp_compress", False):
            model_name = "frozen_model.pb"
        else:
            model_name = "frozen_model_compressed.pb"
        task_file = os.path.join(train_task_fmt % ii, model_name)
        ofile = os.path.join(work_path, "graph.%03d.pb" % ii)
        if os.path.isfile(ofile):
            os.remove(ofile)
        os.symlink(task_file, ofile)

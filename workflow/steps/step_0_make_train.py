# step_0_make_train.py
# created by Yichao


# region
import os
import glob
import json
import time

import dpdata
import logging
import numpy as np
import random
import sys

from utils.utils import copy_model
from packaging.version import Version

from dpgen.generator.lib.utils import (
    create_path,
    make_iter_name,
    symlink_user_forward_files,
    log_task
)

from config.config import *

from dpgen.util import convert_training_data_to_hdf5, expand_sys_str


# endregion





def make_train(iter_index, jdata, mdata):
    # load json param
    # train_param = jdata['train_param']
    train_input_file = default_train_input_file
    numb_models = jdata["numb_models"]
    init_data_prefix = jdata["init_data_prefix"]
    init_data_prefix = os.path.abspath(init_data_prefix)
    init_data_sys_ = jdata["init_data_sys"]
    fp_task_min = jdata["fp_task_min"]
    model_devi_jobs = jdata["model_devi_jobs"]
    use_ele_temp = jdata.get("use_ele_temp", 0)
    training_iter0_model = jdata.get("training_iter0_model_path", [])
    training_init_model = jdata.get("training_init_model", False)
    training_reuse_iter = jdata.get("training_reuse_iter")
    training_reuse_old_ratio = jdata.get("training_reuse_old_ratio", None)

    # if you want to use DP-ZBL potential , you have to give the path of your energy potential file
    if "srtab_file_path" in jdata.keys():
        srtab_file_path = os.path.abspath(jdata.get("srtab_file_path", None))

    if "training_reuse_stop_batch" in jdata.keys():
        training_reuse_stop_batch = jdata["training_reuse_stop_batch"]
    elif "training_reuse_numb_steps" in jdata.keys():
        training_reuse_stop_batch = jdata["training_reuse_numb_steps"]
    else:
        training_reuse_stop_batch = 400000

    training_reuse_start_lr = jdata.get("training_reuse_start_lr", 1e-4)
    training_reuse_start_pref_e = jdata.get("training_reuse_start_pref_e", 0.1)
    training_reuse_start_pref_f = jdata.get("training_reuse_start_pref_f", 100)
    model_devi_activation_func = jdata.get("model_devi_activation_func", None)

    if training_reuse_iter is not None and training_reuse_old_ratio is None:
        raise RuntimeError(
            "training_reuse_old_ratio not found but is mandatory when using init-model (training_reuse_iter is detected in param).\n"
            "It defines the ratio of the old-data picking probability to the all-data(old-data plus new-data) picking probability in training after training_reuse_iter.\n"
            "Denoting the index of the current iter as N (N >= training_reuse_iter ), old-data refers to those existed before the N-1 iter, and new-data refers to that obtained by the N-1 iter.\n"
            "A recommended strategy is making the new-to-old ratio close to 10 times of the default value, to reasonably increase the sensitivity of the model to the new-data.\n"
            "By default, the picking probability of data from one system or one iter is proportional to the number of batches (the number of frames divided by batch_size) of that systems or iter.\n"
            "Detailed discussion about init-model (in Chinese) please see https://mp.weixin.qq.com/s/qsKMZ0j270YhQKvwXUiFvQ"
        )

    model_devi_engine = jdata.get("model_devi_engine", "lammps")
    if iter_index > 0 and _check_empty_iter(iter_index - 1, fp_task_min):
        log_task("prev data is empty, copy prev model")
        copy_model(numb_models, iter_index - 1, iter_index)
        return
    elif (
        model_devi_engine != "calypso"
        and iter_index > 0
        and _check_skip_train(model_devi_jobs[iter_index - 1])
    ):
        log_task("skip training at step %d " % (iter_index - 1))
        copy_model(numb_models, iter_index - 1, iter_index)
        return
    else:
        iter_name = make_iter_name(iter_index)
        work_path = os.path.join(iter_name, train_name)
        copy_flag = os.path.join(work_path, "copied")
        if os.path.isfile(copy_flag):
            os.remove(copy_flag)

    # establish work path
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, train_name)
    create_path(work_path)
    # link init data
    cwd = os.getcwd()
    os.chdir(work_path)
    os.symlink(os.path.abspath(init_data_prefix), "data.init")
    # link iter data
    os.mkdir("data.iters")
    os.chdir("data.iters")
    for ii in range(iter_index):
        os.symlink(
            os.path.relpath(os.path.join(cwd, make_iter_name(ii))), make_iter_name(ii)
        )
    os.chdir(cwd)

    init_data_sys = []
    init_batch_size = []
    if "init_batch_size" in jdata:
        init_batch_size_ = list(jdata["init_batch_size"])
        if len(init_data_sys_) > len(init_batch_size_):
            warnings.warn(
                "The batch sizes are not enough. Assume auto for those not spefified."
            )
            init_batch_size.extend(
                ["auto" for aa in range(len(init_data_sys_) - len(init_batch_size))]
            )
    else:
        init_batch_size_ = ["auto" for aa in range(len(jdata["init_data_sys"]))]
    if "sys_batch_size" in jdata:
        sys_batch_size = jdata["sys_batch_size"]
    else:
        sys_batch_size = ["auto" for aa in range(len(jdata["sys_configs"]))]

    # make sure all init_data_sys has the batch size -- for the following `zip`
    assert len(init_data_sys_) <= len(init_batch_size_)
    for ii, ss in zip(init_data_sys_, init_batch_size_):
        sys_paths = expand_sys_str(os.path.join(init_data_prefix, ii))
        for single_sys in sys_paths:
            init_data_sys.append(
                os.path.normpath(
                    os.path.join(
                        "..",
                        "data.init",
                        ii,
                        os.path.relpath(single_sys, os.path.join(init_data_prefix, ii)),
                    )
                )
            )
            init_batch_size.append(detect_batch_size(ss, single_sys))
    old_range = None
    if iter_index > 0:
        for ii in range(iter_index):
            if ii == iter_index - 1:
                old_range = len(init_data_sys)
            fp_path = os.path.join(make_iter_name(ii), fp_name)
            fp_data_sys = glob.glob(os.path.join(fp_path, "data.*"))
            if model_devi_engine == "calypso":
                _modd_path = os.path.join(
                    make_iter_name(ii), model_devi_name, calypso_model_devi_name
                )
                sys_list = glob.glob(os.path.join(_modd_path, "*.structures"))
                sys_batch_size = ["auto" for aa in range(len(sys_list))]
            for jj in fp_data_sys:
                sys_idx = int(jj.split(".")[-1])
                sys_paths = expand_sys_str(jj)
                nframes = 0
                for sys_single in sys_paths:
                    nframes += dpdata.LabeledSystem(
                        sys_single, fmt="deepmd/npy"
                    ).get_nframes()
                if nframes < fp_task_min:
                    log_task(
                        "nframes (%d) in data sys %s is too small, skip" % (nframes, jj)
                    )
                    continue
                for sys_single in sys_paths:
                    init_data_sys.append(
                        os.path.normpath(os.path.join("..", "data.iters", sys_single))
                    )
                    init_batch_size.append(
                        detect_batch_size(sys_batch_size[sys_idx], sys_single)
                    )
    # establish tasks
    jinput = jdata["default_training_param"]
    try:
        mdata["deepmd_version"]
    except KeyError:
        mdata = set_version(mdata)
    # setup data systems
    if Version(mdata["deepmd_version"]) >= Version("1") and Version(
        mdata["deepmd_version"]
    ) < Version("2"):
        # 1.x
        jinput["training"]["systems"] = init_data_sys
        jinput["training"]["batch_size"] = init_batch_size
        jinput["model"]["type_map"] = jdata["type_map"]
        # electron temperature
        if use_ele_temp == 0:
            pass
        elif use_ele_temp == 1:
            jinput["model"]["fitting_net"]["numb_fparam"] = 1
            jinput["model"]["fitting_net"].pop("numb_aparam", None)
        elif use_ele_temp == 2:
            jinput["model"]["fitting_net"]["numb_aparam"] = 1
            jinput["model"]["fitting_net"].pop("numb_fparam", None)
        else:
            raise RuntimeError("invalid setting for use_ele_temp " + str(use_ele_temp))
    elif Version(mdata["deepmd_version"]) >= Version("2") and Version(
        mdata["deepmd_version"]
    ) < Version("3"):
        # 2.x
        jinput["training"]["training_data"] = {}
        jinput["training"]["training_data"]["systems"] = init_data_sys
        jinput["training"]["training_data"]["batch_size"] = init_batch_size
        jinput["model"]["type_map"] = jdata["type_map"]
        # electron temperature
        if use_ele_temp == 0:
            pass
        elif use_ele_temp == 1:
            jinput["model"]["fitting_net"]["numb_fparam"] = 1
            jinput["model"]["fitting_net"].pop("numb_aparam", None)
        elif use_ele_temp == 2:
            jinput["model"]["fitting_net"]["numb_aparam"] = 1
            jinput["model"]["fitting_net"].pop("numb_fparam", None)
        else:
            raise RuntimeError("invalid setting for use_ele_temp " + str(use_ele_temp))
    else:
        raise RuntimeError(
            "DP-GEN currently only supports for DeePMD-kit 1.x or 2.x version!"
        )
    # set training reuse model
    if training_reuse_iter is not None and iter_index >= training_reuse_iter:
        if "numb_steps" in jinput["training"] and training_reuse_stop_batch is not None:
            jinput["training"]["numb_steps"] = training_reuse_stop_batch
        elif (
            "stop_batch" in jinput["training"] and training_reuse_stop_batch is not None
        ):
            jinput["training"]["stop_batch"] = training_reuse_stop_batch
        if Version("1") <= Version(mdata["deepmd_version"]) < Version("2"):
            jinput["training"][
                "auto_prob_style"
            ] = "prob_sys_size; 0:%d:%f; %d:%d:%f" % (
                old_range,
                training_reuse_old_ratio,
                old_range,
                len(init_data_sys),
                1.0 - training_reuse_old_ratio,
            )
        elif Version("2") <= Version(mdata["deepmd_version"]) < Version("3"):
            jinput["training"]["training_data"][
                "auto_prob"
            ] = "prob_sys_size; 0:%d:%f; %d:%d:%f" % (
                old_range,
                training_reuse_old_ratio,
                old_range,
                len(init_data_sys),
                1.0 - training_reuse_old_ratio,
            )
        else:
            raise RuntimeError(
                "Unsupported DeePMD-kit version: %s" % mdata["deepmd_version"]
            )
        if jinput["loss"].get("start_pref_e") is not None:
            jinput["loss"]["start_pref_e"] = training_reuse_start_pref_e
        if jinput["loss"].get("start_pref_f") is not None:
            jinput["loss"]["start_pref_f"] = training_reuse_start_pref_f
        jinput["learning_rate"]["start_lr"] = training_reuse_start_lr

    input_files = []
    for ii in range(numb_models):
        task_path = os.path.join(work_path, train_task_fmt % ii)
        create_path(task_path)
        os.chdir(task_path)

        if "srtab_file_path" in jdata.keys():
            shutil.copyfile(srtab_file_path, os.path.basename(srtab_file_path))

        for jj in init_data_sys:
            # HDF5 path contains #
            if not (
                os.path.isdir(jj) if "#" not in jj else os.path.isfile(jj.split("#")[0])
            ):
                raise RuntimeError(
                    "data sys %s does not exists, cwd is %s" % (jj, os.getcwd())
                )
        os.chdir(cwd)
        # set random seed for each model
        if Version(mdata["deepmd_version"]) >= Version("1") and Version(
            mdata["deepmd_version"]
        ) < Version("3"):
            # 1.x
            if jinput["model"]["descriptor"]["type"] == "hybrid":
                for desc in jinput["model"]["descriptor"]["list"]:
                    desc["seed"] = random.randrange(sys.maxsize) % (2**32)
            elif jinput["model"]["descriptor"]["type"] == "loc_frame":
                pass
            else:
                jinput["model"]["descriptor"]["seed"] = random.randrange(
                    sys.maxsize
                ) % (2**32)
            jinput["model"]["fitting_net"]["seed"] = random.randrange(sys.maxsize) % (
                2**32
            )
            if "type_embedding" in jinput["model"]:
                jinput["model"]["type_embedding"]["seed"] = random.randrange(
                    sys.maxsize
                ) % (2**32)
            jinput["training"]["seed"] = random.randrange(sys.maxsize) % (2**32)
        else:
            raise RuntimeError(
                "DP-GEN currently only supports for DeePMD-kit 1.x or 2.x version!"
            )
        # set model activation function
        if model_devi_activation_func is not None:
            if Version(mdata["deepmd_version"]) < Version("1"):
                raise RuntimeError(
                    "model_devi_activation_func does not suppport deepmd version",
                    mdata["deepmd_version"],
                )
            assert (
                type(model_devi_activation_func) is list
                and len(model_devi_activation_func) == numb_models
            )
            if (
                len(np.array(model_devi_activation_func).shape) == 2
            ):  # 2-dim list for emd/fitting net-resolved assignment of actF
                jinput["model"]["descriptor"][
                    "activation_function"
                ] = model_devi_activation_func[ii][0]
                jinput["model"]["fitting_net"][
                    "activation_function"
                ] = model_devi_activation_func[ii][1]
            if (
                len(np.array(model_devi_activation_func).shape) == 1
            ):  # for backward compatibility, 1-dim list, not net-resolved
                jinput["model"]["descriptor"][
                    "activation_function"
                ] = model_devi_activation_func[ii]
                jinput["model"]["fitting_net"][
                    "activation_function"
                ] = model_devi_activation_func[ii]
        # dump the input.json
        with open(os.path.join(task_path, train_input_file), "w") as outfile:
            json.dump(jinput, outfile, indent=4)
        input_files.append(os.path.join(task_path, train_input_file))

    # link old models
    if iter_index > 0:
        prev_iter_name = make_iter_name(iter_index - 1)
        prev_work_path = os.path.join(prev_iter_name, train_name)
        for ii in range(numb_models):
            prev_task_path = os.path.join(prev_work_path, train_task_fmt % ii)
            old_model_files = glob.glob(os.path.join(prev_task_path, "model.ckpt*"))
            _link_old_models(work_path, old_model_files, ii)
    else:
        if type(training_iter0_model) == str:
            training_iter0_model = [training_iter0_model]
        iter0_models = []
        for ii in training_iter0_model:
            model_is = glob.glob(ii)
            model_is.sort()
            iter0_models += [os.path.abspath(ii) for ii in model_is]
        if training_init_model:
            assert numb_models == len(iter0_models), (
                "training_iter0_model should be provided, and the number of models should be equal to %d"
                % numb_models
            )
        for ii in range(len(iter0_models)):
            old_model_files = glob.glob(os.path.join(iter0_models[ii], "model.ckpt*"))
            _link_old_models(work_path, old_model_files, ii)
    # Copy user defined forward files
    symlink_user_forward_files(mdata=mdata, task_type="train", work_path=work_path)
    # HDF5 format for training data
    if jdata.get("one_h5", False):
        convert_training_data_to_hdf5(input_files, os.path.join(work_path, "data.hdf5"))


def detect_batch_size(batch_size, system=None):
    if type(batch_size) == int:
        return batch_size
    elif batch_size == "auto":
        # automaticcaly set batch size, batch_size = 32 // atom_numb (>=1, <=fram_numb)
        # check if h5 file
        format = "deepmd/npy" if "#" not in system else "deepmd/hdf5"
        s = dpdata.LabeledSystem(system, fmt=format)
        return int(
            min(np.ceil(32.0 / float(s["coords"].shape[1])), s["coords"].shape[0])
        )
    else:
        raise RuntimeError("Unsupported batch size")



def create_train_workspace(iter_index, jdata, mdata):
    """
    Create workspace for training at given iteration.

    Args:
        iter_index (int): Current iteration index.
        jdata (dict): Job configuration dictionary.
        mdata (dict): Machine configuration dictionary.

    Returns:
        str: Path to the created workspace.
    """
    # Generate timestamp for current iteration
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logging.info(f"Generated timestamp for current iteration: {timestamp}")

    # Create output directory for current iteration
    output_dir = os.path.join("output", "tasks", timestamp)
    logging.info(f"Creating output directory for current iteration: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Generate iteration name
    iter_name = make_iter_name(iter_index)
    logging.info(f"Generated name for current iteration: {iter_name}")

    # Generate work path for current iteration
    work_path = os.path.join(output_dir, iter_name, train_name)
    logging.info(f"Generated work path for current iteration training: {work_path}")

    # Create work path
    create_path(work_path)
    logging.info(f"Created work path for current iteration training: {work_path}")

    # Check and remove 'copied' flag if exists
    copy_flag = os.path.join(work_path, "copied")
    if os.path.isfile(copy_flag):
        logging.info(f"Found 'copied' flag in {work_path}, removing it")
        os.remove(copy_flag)

    # Link init data
    cwd = os.getcwd()
    os.chdir(work_path)
    # os.symlink(os.path.abspath(jdata["init_data_prefix"]), "data.init")

    init_data_dir = os.path.abspath(jdata["init_data_prefix"])
    if init_data_dir != work_path:
        os.symlink(init_data_dir, "data.init")
    else:
        logging.warning(f"Init data prefix is the same as work path, skipping symlink creation")


    # Link iter data
    logging.info(f"Creating directory 'data.iters' in {work_path}")
    os.mkdir("data.iters")
    logging.info(f"Changing current working directory to {os.path.join(work_path, 'data.iters')}")
    os.chdir("data.iters")
    for ii in range(iter_index):
        os.symlink(os.path.relpath(os.path.join(cwd, make_iter_name(ii))), make_iter_name(ii))
    os.chdir(cwd)

    return work_path

def make_train_legacy(iter_index, jdata, mdata):
    # load json param
    # train_param = jdata['train_param']
    train_input_file = default_train_input_file
    numb_models = jdata["numb_models"]
    init_data_prefix = jdata["init_data_prefix"]
    init_data_prefix = os.path.abspath(init_data_prefix)
    init_data_sys_ = jdata["init_data_sys"]
    fp_task_min = jdata["fp_task_min"]
    model_devi_jobs = jdata["model_devi_jobs"]
    use_ele_temp = jdata.get("use_ele_temp", 0)
    training_iter0_model = jdata.get("training_iter0_model_path", [])
    training_init_model = jdata.get("training_init_model", False)
    training_reuse_iter = jdata.get("training_reuse_iter")
    training_reuse_old_ratio = jdata.get("training_reuse_old_ratio", None)

    # if you want to use DP-ZBL potential , you have to give the path of your energy potential file
    if "srtab_file_path" in jdata.keys():
        srtab_file_path = os.path.abspath(jdata.get("srtab_file_path", None))

    if "training_reuse_stop_batch" in jdata.keys():
        training_reuse_stop_batch = jdata["training_reuse_stop_batch"]
    elif "training_reuse_numb_steps" in jdata.keys():
        training_reuse_stop_batch = jdata["training_reuse_numb_steps"]
    else:
        training_reuse_stop_batch = 400000

    training_reuse_start_lr = jdata.get("training_reuse_start_lr", 1e-4)
    training_reuse_start_pref_e = jdata.get("training_reuse_start_pref_e", 0.1)
    training_reuse_start_pref_f = jdata.get("training_reuse_start_pref_f", 100)
    model_devi_activation_func = jdata.get("model_devi_activation_func", None)

    if training_reuse_iter is not None and training_reuse_old_ratio is None:
        raise RuntimeError(
            "training_reuse_old_ratio not found but is mandatory when using init-model (training_reuse_iter is detected in param).\n"
            "It defines the ratio of the old-data picking probability to the all-data(old-data plus new-data) picking probability in training after training_reuse_iter.\n"
            "Denoting the index of the current iter as N (N >= training_reuse_iter ), old-data refers to those existed before the N-1 iter, and new-data refers to that obtained by the N-1 iter.\n"
            "A recommended strategy is making the new-to-old ratio close to 10 times of the default value, to reasonably increase the sensitivity of the model to the new-data.\n"
            "By default, the picking probability of data from one system or one iter is proportional to the number of batches (the number of frames divided by batch_size) of that systems or iter.\n"
            "Detailed discussion about init-model (in Chinese) please see https://mp.weixin.qq.com/s/qsKMZ0j270YhQKvwXUiFvQ"
        )

    model_devi_engine = jdata.get("model_devi_engine", "lammps")

    # if iter_index > 0 and _check_empty_iter(iter_index - 1, fp_task_min):
    #     logging.info(f"Previous iteration (iter.{iter_index - 1:06d}) data is empty")
    #     log_task("prev data is empty, copy prev model")
    #     logging.info(
    #         f"Copying models from previous iteration (iter.{iter_index - 1:06d}) to current iteration (iter.{iter_index:06d})")
    #     copy_model(numb_models, iter_index - 1, iter_index)
    #     logging.info(f"Skipping current iteration (iter.{iter_index:06d}) training")
    #     return
    # elif (
    #         model_devi_engine != "calypso"
    #         and iter_index > 0
    #         and _check_skip_train(model_devi_jobs[iter_index - 1])
    # ):
    #     logging.info(f"Training skip condition met for previous iteration (iter.{iter_index - 1:06d})")
    #     log_task("skip training at step %d " % (iter_index - 1))
    #     logging.info(
    #         f"Copying models from previous iteration (iter.{iter_index - 1:06d}) to current iteration (iter.{iter_index:06d})")
    #     copy_model(numb_models, iter_index - 1, iter_index)
    #     logging.info(f"Skipping current iteration (iter.{iter_index:06d}) training")
    #     return
    # else:
    #
    #     # Generate timestamp for current iteration
    #     timestamp = time.strftime("%Y%m%d-%H%M%S")
    #     logging.info(f"Generated timestamp for current iteration: {timestamp}")
    #
    #     # Create output directory for current iteration
    #     output_dir = os.path.join("output", "tasks", timestamp)
    #     logging.info(f"Creating output directory for current iteration: {output_dir}")
    #     os.makedirs(output_dir, exist_ok=True)
    #
    #     # Generate iteration name
    #     logging.info(f"Generating name for current iteration (iter.{iter_index:06d})")
    #     iter_name = make_iter_name(iter_index)
    #     logging.info(f"Generated name for current iteration: {iter_name}")
    #
    #     # Generate work path for current iteration
    #     logging.info(f"Generating work path for current iteration training")
    #     work_path = os.path.join(output_dir, iter_name, train_name)
    #     logging.info(f"Generated work path for current iteration training: {work_path}")
    #
    #     copy_flag = os.path.join(work_path, "copied")
    #     if os.path.isfile(copy_flag):
    #         logging.info(f"Found 'copied' flag in {work_path}, removing it")
    #         os.remove(copy_flag)
    #
    #     logging.info(f"Establishing work path for current iteration training")
    #
    #     logging.info(f"Generating name for current iteration (iter.{iter_index:06d})")
    #     iter_name = make_iter_name(iter_index)
    #     logging.info(f"Generated name for current iteration: {iter_name}")
    #
    #     logging.info(f"Generating work path for current iteration training")
    #     work_path = os.path.join(iter_name, train_name)
    #     logging.info(f"Generated work path for current iteration training: {work_path}")
    #
    #     logging.info(f"Creating work path for current iteration training")
    #     create_path(work_path)
    #     logging.info(f"Created work path for current iteration training: {work_path}")
    # # link init data
    # cwd = os.getcwd()
    # os.chdir(work_path)
    # os.symlink(os.path.abspath(init_data_prefix), "data.init")
    #
    # # link iter data
    # logging.info(f"Creating directory 'data.iters' in {work_path}")
    # os.mkdir("data.iters")
    # logging.info(f"Changing current working directory to {os.path.join(work_path, 'data.iters')}")
    # os.chdir("data.iters")
    #
    # # Check if the created workspace exists
    # if os.path.exists(os.path.join(work_path, f"iter.{iter_index:06d}")):
    #     logging.info(f"Workspace for iteration {iter_index:06d} created successfully at {work_path}")
    # else:
    #     logging.error(f"Failed to create workspace for iteration {iter_index:06d} at {work_path}")
    #     raise RuntimeError(f"Workspace creation failed for iteration {iter_index:06d}")



    if iter_index > 0 and _check_empty_iter(iter_index - 1, jdata["fp_task_min"]):
        logging.info(f"Previous iteration (iter.{iter_index - 1:06d}) data is empty")
        log_task("prev data is empty, copy prev model")
        logging.info(f"Copying models from previous iteration (iter.{iter_index - 1:06d}) to current iteration (iter.{iter_index:06d})")
        copy_model(jdata["numb_models"], iter_index - 1, iter_index)
        logging.info(f"Skipping current iteration (iter.{iter_index:06d}) training")
        return
    elif model_devi_engine != "calypso" and iter_index > 0 and _check_skip_train(jdata["model_devi_jobs"][iter_index - 1]):
        logging.info(f"Training skip condition met for previous iteration (iter.{iter_index - 1:06d})")
        log_task("skip training at step %d" % (iter_index - 1))
        logging.info(f"Copying models from previous iteration (iter.{iter_index - 1:06d}) to current iteration (iter.{iter_index:06d})")
        copy_model(jdata["numb_models"], iter_index - 1, iter_index)
        logging.info(f"Skipping current iteration (iter.{iter_index:06d}) training")
        return
    else:
        work_path = create_train_workspace(iter_index, jdata, mdata)



    # for ii in range(iter_index):
    #     os.symlink(
    #         os.path.relpath(os.path.join(cwd, make_iter_name(ii))), make_iter_name(ii)
    #     )
    # os.chdir(cwd)

    init_data_sys = []
    init_batch_size = []
    if "init_batch_size" in jdata:
        init_batch_size_ = list(jdata["init_batch_size"])
        if len(init_data_sys_) > len(init_batch_size_):
            warnings.warn(
                "The batch sizes are not enough. Assume auto for those not spefified."
            )
            init_batch_size.extend(
                ["auto" for aa in range(len(init_data_sys_) - len(init_batch_size))]
            )
    else:
        init_batch_size_ = ["auto" for aa in range(len(jdata["init_data_sys"]))]
    if "sys_batch_size" in jdata:
        sys_batch_size = jdata["sys_batch_size"]
    else:
        sys_batch_size = ["auto" for aa in range(len(jdata["sys_configs"]))]

    # make sure all init_data_sys has the batch size -- for the following `zip`
    assert len(init_data_sys_) <= len(init_batch_size_)
    for ii, ss in zip(init_data_sys_, init_batch_size_):
        sys_paths = expand_sys_str(os.path.join(init_data_prefix, ii))
        for single_sys in sys_paths:
            init_data_sys.append(
                os.path.normpath(
                    os.path.join(
                        work_path,
                        "data.init",
                        ii,
                        os.path.relpath(single_sys, os.path.join(init_data_prefix, ii)),
                    )
                )
            )
            init_batch_size.append(detect_batch_size(ss, single_sys))
    old_range = None
    if iter_index > 0:
        for ii in range(iter_index):
            if ii == iter_index - 1:
                old_range = len(init_data_sys)
            fp_path = os.path.join(make_iter_name(ii), fp_name)
            fp_data_sys = glob.glob(os.path.join(fp_path, "data.*"))
            if model_devi_engine == "calypso":
                _modd_path = os.path.join(
                    make_iter_name(ii), model_devi_name, calypso_model_devi_name
                )
                sys_list = glob.glob(os.path.join(_modd_path, "*.structures"))
                sys_batch_size = ["auto" for aa in range(len(sys_list))]
            for jj in fp_data_sys:
                sys_idx = int(jj.split(".")[-1])
                sys_paths = expand_sys_str(jj)
                nframes = 0
                for sys_single in sys_paths:
                    nframes += dpdata.LabeledSystem(
                        sys_single, fmt="deepmd/npy"
                    ).get_nframes()
                if nframes < fp_task_min:
                    log_task(
                        "nframes (%d) in data sys %s is too small, skip" % (nframes, jj)
                    )
                    continue
                for sys_single in sys_paths:
                    init_data_sys.append(
                        os.path.normpath(os.path.join("..", "data.iters", sys_single))
                    )
                    init_batch_size.append(
                        detect_batch_size(sys_batch_size[sys_idx], sys_single)
                    )
    # establish tasks
    jinput = jdata["default_training_param"]
    try:
        mdata["deepmd_version"]
    except KeyError:
        mdata = set_version(mdata)
    # setup data systems
    if Version(mdata["deepmd_version"]) >= Version("1") and Version(
            mdata["deepmd_version"]
    ) < Version("2"):
        # 1.x
        jinput["training"]["systems"] = init_data_sys
        jinput["training"]["batch_size"] = init_batch_size
        jinput["model"]["type_map"] = jdata["type_map"]
        # electron temperature
        if use_ele_temp == 0:
            pass
        elif use_ele_temp == 1:
            jinput["model"]["fitting_net"]["numb_fparam"] = 1
            jinput["model"]["fitting_net"].pop("numb_aparam", None)
        elif use_ele_temp == 2:
            jinput["model"]["fitting_net"]["numb_aparam"] = 1
            jinput["model"]["fitting_net"].pop("numb_fparam", None)
        else:
            raise RuntimeError("invalid setting for use_ele_temp " + str(use_ele_temp))
    elif Version(mdata["deepmd_version"]) >= Version("2") and Version(
            mdata["deepmd_version"]
    ) < Version("3"):
        # 2.x
        jinput["training"]["training_data"] = {}
        jinput["training"]["training_data"]["systems"] = init_data_sys
        jinput["training"]["training_data"]["batch_size"] = init_batch_size
        jinput["model"]["type_map"] = jdata["type_map"]
        # electron temperature
        if use_ele_temp == 0:
            pass
        elif use_ele_temp == 1:
            jinput["model"]["fitting_net"]["numb_fparam"] = 1
            jinput["model"]["fitting_net"].pop("numb_aparam", None)
        elif use_ele_temp == 2:
            jinput["model"]["fitting_net"]["numb_aparam"] = 1
            jinput["model"]["fitting_net"].pop("numb_fparam", None)
        else:
            raise RuntimeError("invalid setting for use_ele_temp " + str(use_ele_temp))
    else:
        raise RuntimeError(
            "DP-GEN currently only supports for DeePMD-kit 1.x or 2.x version!"
        )
    # set training reuse model
    if training_reuse_iter is not None and iter_index >= training_reuse_iter:
        if "numb_steps" in jinput["training"] and training_reuse_stop_batch is not None:
            jinput["training"]["numb_steps"] = training_reuse_stop_batch
        elif (
                "stop_batch" in jinput["training"] and training_reuse_stop_batch is not None
        ):
            jinput["training"]["stop_batch"] = training_reuse_stop_batch
        if Version("1") <= Version(mdata["deepmd_version"]) < Version("2"):
            jinput["training"][
                "auto_prob_style"
            ] = "prob_sys_size; 0:%d:%f; %d:%d:%f" % (
                old_range,
                training_reuse_old_ratio,
                old_range,
                len(init_data_sys),
                1.0 - training_reuse_old_ratio,
            )
        elif Version("2") <= Version(mdata["deepmd_version"]) < Version("3"):
            jinput["training"]["training_data"][
                "auto_prob"
            ] = "prob_sys_size; 0:%d:%f; %d:%d:%f" % (
                old_range,
                training_reuse_old_ratio,
                old_range,
                len(init_data_sys),
                1.0 - training_reuse_old_ratio,
            )
        else:
            raise RuntimeError(
                "Unsupported DeePMD-kit version: %s" % mdata["deepmd_version"]
            )
        if jinput["loss"].get("start_pref_e") is not None:
            jinput["loss"]["start_pref_e"] = training_reuse_start_pref_e
        if jinput["loss"].get("start_pref_f") is not None:
            jinput["loss"]["start_pref_f"] = training_reuse_start_pref_f
        jinput["learning_rate"]["start_lr"] = training_reuse_start_lr

    input_files = []
    for ii in range(numb_models):
        task_path = os.path.join(work_path, train_task_fmt % ii)
        create_path(task_path)
        os.chdir(task_path)

        if "srtab_file_path" in jdata.keys():
            shutil.copyfile(srtab_file_path, os.path.basename(srtab_file_path))

        for jj in init_data_sys:
            jj = os.path.abspath(jj)
            # HDF5 path contains #
            if not (
                    os.path.isdir(jj) if "#" not in jj else os.path.isfile(jj.split("#")[0])
            ):
                raise RuntimeError(
                    "data sys %s does not exists, cwd is %s" % (jj, os.getcwd())
                )
        os.chdir(cwd)
        # set random seed for each model
        if Version(mdata["deepmd_version"]) >= Version("1") and Version(
                mdata["deepmd_version"]
        ) < Version("3"):
            # 1.x
            if jinput["model"]["descriptor"]["type"] == "hybrid":
                for desc in jinput["model"]["descriptor"]["list"]:
                    desc["seed"] = random.randrange(sys.maxsize) % (2 ** 32)
            elif jinput["model"]["descriptor"]["type"] == "loc_frame":
                pass
            else:
                jinput["model"]["descriptor"]["seed"] = random.randrange(
                    sys.maxsize
                ) % (2 ** 32)
            jinput["model"]["fitting_net"]["seed"] = random.randrange(sys.maxsize) % (
                    2 ** 32
            )
            if "type_embedding" in jinput["model"]:
                jinput["model"]["type_embedding"]["seed"] = random.randrange(
                    sys.maxsize
                ) % (2 ** 32)
            jinput["training"]["seed"] = random.randrange(sys.maxsize) % (2 ** 32)
        else:
            raise RuntimeError(
                "DP-GEN currently only supports for DeePMD-kit 1.x or 2.x version!"
            )
        # set model activation function
        if model_devi_activation_func is not None:
            if Version(mdata["deepmd_version"]) < Version("1"):
                raise RuntimeError(
                    "model_devi_activation_func does not suppport deepmd version",
                    mdata["deepmd_version"],
                )
            assert (
                    type(model_devi_activation_func) is list
                    and len(model_devi_activation_func) == numb_models
            )
            if (
                    len(np.array(model_devi_activation_func).shape) == 2
            ):  # 2-dim list for emd/fitting net-resolved assignment of actF
                jinput["model"]["descriptor"][
                    "activation_function"
                ] = model_devi_activation_func[ii][0]
                jinput["model"]["fitting_net"][
                    "activation_function"
                ] = model_devi_activation_func[ii][1]
            if (
                    len(np.array(model_devi_activation_func).shape) == 1
            ):  # for backward compatibility, 1-dim list, not net-resolved
                jinput["model"]["descriptor"][
                    "activation_function"
                ] = model_devi_activation_func[ii]
                jinput["model"]["fitting_net"][
                    "activation_function"
                ] = model_devi_activation_func[ii]
        # dump the input.json
        with open(os.path.join(task_path, train_input_file), "w") as outfile:
            json.dump(jinput, outfile, indent=4)
        input_files.append(os.path.join(task_path, train_input_file))

    # link old models
    if iter_index > 0:
        prev_iter_name = make_iter_name(iter_index - 1)
        prev_work_path = os.path.join(prev_iter_name, train_name)
        for ii in range(numb_models):
            prev_task_path = os.path.join(prev_work_path, train_task_fmt % ii)
            old_model_files = glob.glob(os.path.join(prev_task_path, "model.ckpt*"))
            _link_old_models(work_path, old_model_files, ii)
    else:
        if type(training_iter0_model) == str:
            training_iter0_model = [training_iter0_model]
        iter0_models = []
        for ii in training_iter0_model:
            model_is = glob.glob(ii)
            model_is.sort()
            iter0_models += [os.path.abspath(ii) for ii in model_is]
        if training_init_model:
            assert numb_models == len(iter0_models), (
                    "training_iter0_model should be provided, and the number of models should be equal to %d"
                    % numb_models
            )
        for ii in range(len(iter0_models)):
            old_model_files = glob.glob(os.path.join(iter0_models[ii], "model.ckpt*"))
            _link_old_models(work_path, old_model_files, ii)
    # Copy user defined forward files
    symlink_user_forward_files(mdata=mdata, task_type="train", work_path=work_path)
    # HDF5 format for training data
    if jdata.get("one_h5", False):
        convert_training_data_to_hdf5(input_files, os.path.join(work_path, "data.hdf5"))

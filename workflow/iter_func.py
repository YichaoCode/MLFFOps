# iter_func.py
# created by Yichao

# region
import os
import glob
import json
import dpdata
import logging
import numpy as np
import random
import sys
import itertools

from packaging.version import Version 
from dpgen.generator.lib.run_calypso import run_calypso_model_devi

from packaging.version import Version


from dpgen.generator.lib.utils import (
    create_path,
    make_iter_name,
    symlink_user_forward_files,
    log_task
)

from workflow.lammps_run import run_lammps

from fp_interface.fp_vasp import make_fp_vasp, post_fp_vasp, _vasp_check_fin
from fp_interface.fp_pwscf import make_fp_pwscf, post_fp_pwscf  
from fp_interface.fp_siesta import make_fp_siesta, post_fp_siesta
from fp_interface.fp_gaussian import make_fp_gaussian, post_fp_gaussian
from fp_interface.fp_cp2k import make_fp_cp2k, post_fp_cp2k


from dpgen.dispatcher.Dispatcher import make_submission


from model_devi import (
    _make_model_devi_native, _make_model_devi_native_gromacs,
    _make_model_devi_amber, _make_model_devi_revmat
)

from config.config import *


from utils.utils import (
    expand_idx,
    make_model_devi_conf_name
)

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



def run_train_old(iter_index, jdata, mdata):
    # print("debug:run_train:mdata", mdata)
    # load json param
    numb_models = jdata["numb_models"]
    # train_param = jdata['train_param']
    train_input_file = default_train_input_file
    training_reuse_iter = jdata.get("training_reuse_iter")
    training_init_model = jdata.get("training_init_model", False)

    if "srtab_file_path" in jdata.keys():
        zbl_file = os.path.basename(jdata.get("srtab_file_path", None))

    if training_reuse_iter is not None and iter_index >= training_reuse_iter:
        training_init_model = True
    try:
        mdata["deepmd_version"]
    except KeyError:
        mdata = set_version(mdata)

    train_command = mdata.get("train_command", "dp")
    train_resources = mdata["train_resources"]

    # paths
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, train_name)
    # check if is copied
    copy_flag = os.path.join(work_path, "copied")
    if os.path.isfile(copy_flag):
        log_task("copied model, do not train")
        return
    # make tasks
    all_task = []
    for ii in range(numb_models):
        task_path = os.path.join(work_path, train_task_fmt % ii)
        all_task.append(task_path)
    commands = []
    if Version(mdata["deepmd_version"]) >= Version("1") and Version(
        mdata["deepmd_version"]
    ) < Version("3"):

        # 1.x
        ## Commands are like `dp train` and `dp freeze`
        ## train_command should not be None
        assert train_command
        command = train_command
      # # if training_init_model:
       #     command = (
      #          "{ if [ ! -f model.ckpt.index ]; then %s --init-model old/model.ckpt; else %s --restart model.ckpt; fi }"
      #          % (command, command)
     #       )
      #  else:
       #     command = (
      #          "{ if [ ! -f model.ckpt.index ]; then %s; else %s --restart model.ckpt; fi }"
      #          % (command, command)
      #      )
       # command = "/bin/sh -c '%s'" % command
       # commands.append(command)
        command = "%s ; dp freeze " % train_command
        commands.append(command)
        if jdata.get("dp_compress", False):
            commands.append("%s compress" % train_command)
    else:
        raise RuntimeError(
            "DP-GEN currently only supports for DeePMD-kit 1.x or 2.x version!"
        )

    # _tasks = [os.path.basename(ii) for ii in all_task]
    # run_tasks = []
    # for ii in all_task:
    #     check_pb = os.path.join(ii, "frozen_model.pb")
    #     check_lcurve = os.path.join(ii, "lcurve.out")
    #     if os.path.isfile(check_pb) and os.path.isfile(check_lcurve):
    #         pass
    #     else:
    #         run_tasks.append(ii)
    run_tasks = [os.path.basename(ii) for ii in all_task]

    forward_files = [train_input_file]
    if "srtab_file_path" in jdata.keys():
        forward_files.append(zbl_file)
    if training_init_model:
        forward_files += [
            os.path.join("old", "model.ckpt.meta"),
            os.path.join("old", "model.ckpt.index"),
            os.path.join("old", "model.ckpt.data-00000-of-00001"),
        ]
    backward_files = ["frozen_model.pb", "lcurve.out", "train.log"]
    backward_files += [
        "model.ckpt.meta",
        "model.ckpt.index",
        "model.ckpt.data-00000-of-00001",
        "checkpoint",
    ]
    if jdata.get("dp_compress", False):
        backward_files.append("frozen_model_compressed.pb")
    if not jdata.get("one_h5", False):
        init_data_sys_ = jdata["init_data_sys"]
        init_data_sys = []
        for ii in init_data_sys_:
            init_data_sys.append(os.path.join("data.init", ii))
        trans_comm_data = []
        cwd = os.getcwd()
        os.chdir(work_path)
        fp_data = glob.glob(os.path.join("data.iters", "iter.*", "02.fp", "data.*"))
        for ii in itertools.chain(init_data_sys, fp_data):
            sys_paths = expand_sys_str(ii)
            for single_sys in sys_paths:
                if "#" not in single_sys:
                    trans_comm_data += glob.glob(os.path.join(single_sys, "set.*"))
                    trans_comm_data += glob.glob(os.path.join(single_sys, "type*.raw"))
                    trans_comm_data += glob.glob(os.path.join(single_sys, "nopbc"))
                else:
                    # H5 file
                    trans_comm_data.append(single_sys.split("#")[0])
    else:
        cwd = os.getcwd()
        trans_comm_data = ["data.hdf5"]
    # remove duplicated files
    trans_comm_data = list(set(trans_comm_data))
    os.chdir(cwd)

    try:
        train_group_size = mdata["train_group_size"]
    except Exception:
        train_group_size = 1

    api_version = mdata.get("api_version", "1.0")

    user_forward_files = mdata.get("train" + "_user_forward_files", [])
    forward_files += [os.path.basename(file) for file in user_forward_files]
    backward_files += mdata.get("train" + "_user_backward_files", [])
    if Version(api_version) < Version("1.0"):
        raise RuntimeError(
            "API version %s has been removed. Please upgrade to 1.0." % api_version
        )

    elif Version(api_version) >= Version("1.0"):
        submission = make_submission(
            mdata["train_machine"],
            mdata["train_resources"],
            commands=commands,
            work_path=work_path,
            run_tasks=run_tasks,
            group_size=train_group_size,
            forward_common_files=trans_comm_data,
            forward_files=forward_files,
            backward_files=backward_files,
            outlog="train.log",
            errlog="train.log",
        )
        submission.run_submission()

def run_train(iter_index, jdata, mdata):
    try:
        # load json param
        numb_models = jdata["numb_models"]
        train_input_file = default_train_input_file
        training_reuse_iter = jdata.get("training_reuse_iter")
        training_init_model = jdata.get("training_init_model", False)
        
        logging.debug("Loaded parameters from jdata: numb_models=%s, train_input_file=%s, training_reuse_iter=%s, training_init_model=%s", 
            numb_models, train_input_file, training_reuse_iter, training_init_model)
        
        if "srtab_file_path" in jdata.keys():
            zbl_file = os.path.basename(jdata.get("srtab_file_path", None))
            logging.debug("ZBL file path found: %s", zbl_file)

        if training_reuse_iter is not None and iter_index >= training_reuse_iter:
            training_init_model = True
        
        mdata = set_version(mdata) if "deepmd_version" not in mdata.keys() else mdata
        logging.debug("DeePMD version: %s", mdata["deepmd_version"])
        
        train_command = mdata.get("train_command", "dp")
        train_resources = mdata["train_resources"]
        logging.info("Training command: %s", train_command)

        # paths
        iter_name = make_iter_name(iter_index)
        work_path = os.path.join(iter_name, train_name)
        logging.debug("Work path: %s", work_path)
        
        # check if is copied
        copy_flag = os.path.join(work_path, "copied")
        if os.path.isfile(copy_flag):
            logging.info("Copied model, do not train")
            return
        
        # make tasks  
        all_task = [os.path.join(work_path, train_task_fmt % ii) for ii in range(numb_models)]
        logging.debug("Generated %d task paths", len(all_task))
        
        commands = []
        dp_version = Version(mdata["deepmd_version"])
        if dp_version >= Version("1") and dp_version < Version("3"):
            command = "%s ; dp freeze " % train_command
            commands.append(command)
            if jdata.get("dp_compress", False):
                commands.append("%s compress" % train_command)
        else:
            raise RuntimeError("DP-GEN currently only supports for DeePMD-kit 1.x or 2.x version!")
        
        run_tasks = [os.path.basename(ii) for ii in all_task]
        logging.info("Tasks to run: %s", " ".join(run_tasks))
        
        forward_files = [train_input_file]
        if "srtab_file_path" in jdata.keys():
            forward_files.append(zbl_file)
        if training_init_model:
            forward_files += [
                os.path.join("old", "model.ckpt.meta"),
                os.path.join("old", "model.ckpt.index"), 
                os.path.join("old", "model.ckpt.data-00000-of-00001")
            ]
        logging.debug("Forward files: %s", forward_files)
            
        backward_files = ["frozen_model.pb", "lcurve.out", "train.log"]    
        backward_files += [
            "model.ckpt.meta",
            "model.ckpt.index",
            "model.ckpt.data-00000-of-00001",
            "checkpoint"
        ]
        if jdata.get("dp_compress", False):
            backward_files.append("frozen_model_compressed.pb")
        logging.debug("Backward files: %s", backward_files)
            
        if not jdata.get("one_h5", False):
            init_data_sys_ = jdata["init_data_sys"]
            init_data_sys = ["data.init/"+ii for ii in init_data_sys_]
            cwd = os.getcwd()
            os.chdir(work_path)
            fp_data = glob.glob(os.path.join("data.iters", "iter.*", "02.fp", "data.*"))
            trans_comm_data = []
            for ii in init_data_sys + fp_data:
                logging.debug("Processing data file: %s", ii)
                sys_paths = expand_sys_str(ii)
                for single_sys in sys_paths:
                    if "#" not in single_sys:
                        trans_comm_data += glob.glob(os.path.join(single_sys, "set.*"))
                        trans_comm_data += glob.glob(os.path.join(single_sys, "type*.raw"))
                        trans_comm_data += glob.glob(os.path.join(single_sys, "nopbc"))
                    else:
                        # H5 file
                        trans_comm_data.append(single_sys.split("#")[0])
            os.chdir(cwd)
        else:
            trans_comm_data = ["data.hdf5"]
            logging.debug("Using single HDF5 file: %s", trans_comm_data)
        
        # remove duplicated files 
        trans_comm_data = list(set(trans_comm_data))
        logging.debug("Unique common data files: %s", trans_comm_data)
        
        train_group_size = mdata.get("train_group_size", 1)
        logging.info("Train group size: %d", train_group_size)
        
        api_version = mdata.get("api_version", "1.0")
        if Version(api_version) < Version("1.0"):
            raise RuntimeError("API version %s has been removed. Please upgrade to 1.0." % api_version)

        user_forward_files  = [os.path.basename(file) for file in mdata.get("train_user_forward_files", [])]
        user_backward_files = mdata.get("train_user_backward_files", [])
        forward_files  += user_forward_files
        backward_files += user_backward_files
        logging.debug("User forward files: %s", user_forward_files)   
        logging.debug("User backward files: %s", user_backward_files)

        submission = make_submission(
            mdata["train_machine"],
            mdata["train_resources"],
            commands=commands,
            work_path=work_path,
            run_tasks=run_tasks,
            group_size=train_group_size,
            forward_common_files=trans_comm_data,
            forward_files=forward_files,
            backward_files=backward_files,
            outlog="train.log",
            errlog="train.log",
        )
        
        submission.run_submission()
        logging.info("Task submission completed")
        
    except KeyError as e:
        logging.error("Missing key in jdata or mdata: %s", str(e))
        raise
    except RuntimeError as e:
        logging.error("Error occurred: %s", str(e))
        raise
    except Exception as e:
        logging.error("Unexpected error occurred: %s", str(e))
        raise


def post_train(iter_index, jdata, mdata):
    # load json param
    numb_models = jdata["numb_models"]
    # paths
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, train_name)
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



def make_model_devi(iter_index, jdata, mdata):
    # The MD engine to perform model deviation
    # Default is lammps
    model_devi_engine = jdata.get("model_devi_engine", "lammps")

    model_devi_jobs = jdata["model_devi_jobs"]
    if model_devi_engine != "calypso":
        if iter_index >= len(model_devi_jobs):
            return False
    else:
        # mode 1: generate structures according to the user-provided input.dat file, so calypso_input_path and model_devi_max_iter are needed
        run_mode = 1
        if "calypso_input_path" in jdata:
            try:
                maxiter = jdata.get("model_devi_max_iter")
            except KeyError:
                raise KeyError(
                    "calypso_input_path key exists so you should provide model_devi_max_iter key to control the max iter number"
                )
        # mode 2: control each iteration to generate structures in specific way by providing model_devi_jobs key
        else:
            try:
                maxiter = max(model_devi_jobs[-1].get("times"))
                run_mode = 2
            except KeyError:
                raise KeyError('did not find model_devi_jobs["times"] key')
        if iter_index > maxiter:
            dlog.info(f"iter_index is {iter_index} and maxiter is {maxiter}")
            return False

    if "sys_configs_prefix" in jdata:
        sys_configs = []
        for sys_list in jdata["sys_configs"]:
            # assert (isinstance(sys_list, list) ), "Currently only support type list for sys in 'sys_conifgs' "
            temp_sys_list = [
                os.path.join(jdata["sys_configs_prefix"], sys) for sys in sys_list
            ]
            sys_configs.append(temp_sys_list)
    else:
        sys_configs = jdata["sys_configs"]
    shuffle_poscar = jdata.get("shuffle_poscar", False)

    if model_devi_engine != "calypso":
        cur_job = model_devi_jobs[iter_index]
        sys_idx = expand_idx(cur_job["sys_idx"])
    else:
        cur_job = {"model_devi_engine": "calypso", "input.dat": "user_provided"}
        sys_idx = []

    if len(sys_idx) != len(list(set(sys_idx))):
        raise RuntimeError("system index should be uniq")
    conf_systems = []
    for idx in sys_idx:
        cur_systems = []
        ss = sys_configs[idx]
        for ii in ss:
            ii_systems = sorted(glob.glob(ii))
            if ii_systems == []:
                warnings.warn(
                    "There is no system in the path %s. Please check if the path is correct."
                    % ii
                )
            cur_systems += ii_systems
        # cur_systems should not be sorted, as we may add specific constrict to the similutions
        # cur_systems.sort()
        cur_systems = [os.path.abspath(ii) for ii in cur_systems]
        conf_systems.append(cur_systems)

    iter_name = make_iter_name(iter_index)
    train_path = os.path.join(iter_name, TRAIN_NAME)
    train_path = os.path.abspath(train_path)
    models = sorted(glob.glob(os.path.join(train_path, "graph*pb")))
    work_path = os.path.join(iter_name, model_devi_name)
    create_path(work_path)
    if model_devi_engine == "calypso":
        _calypso_run_opt_path = os.path.join(work_path, calypso_run_opt_name)
        calypso_model_devi_path = os.path.join(work_path, calypso_model_devi_name)
        create_path(calypso_model_devi_path)
        # run model devi script
        calypso_run_model_devi_script = os.path.join(
            calypso_model_devi_path, "calypso_run_model_devi.py"
        )
        shutil.copyfile(calypso_run_model_devi_file, calypso_run_model_devi_script)
        # Create work path list
        calypso_run_opt_path = []

        # mode 1: generate structures according to the user-provided input.dat file,
        # so calypso_input_path and model_devi_max_iter are needed
        if run_mode == 1:
            if jdata.get("vsc", False) and len(jdata.get("type_map")) > 1:
                # [input.dat.Li.250, input.dat.Li.300]
                one_ele_inputdat_list = glob.glob(
                    f"{jdata.get('calypso_input_path')}/input.dat.{jdata.get('type_map')[0]}.*"
                )
                if len(one_ele_inputdat_list) == 0:
                    number_of_pressure = 1
                else:
                    number_of_pressure = len(list(set(one_ele_inputdat_list)))

                # calypso_run_opt_path = ['gen_struc_analy.000','gen_struc_analy.001']
                for temp_idx in range(number_of_pressure):
                    calypso_run_opt_path.append(
                        "%s.%03d" % (_calypso_run_opt_path, temp_idx)
                    )
            elif not jdata.get("vsc", False):
                calypso_run_opt_path.append("%s.%03d" % (_calypso_run_opt_path, 0))

        # mode 2: control each iteration to generate structures in specific way
        # by providing model_devi_jobs key
        elif run_mode == 2:
            for iiidx, jobbs in enumerate(model_devi_jobs):
                if iter_index in jobbs.get("times"):
                    cur_job = model_devi_jobs[iiidx]

            pressures_list = cur_job.get("PSTRESS", [0.0001])
            for temp_idx in range(len(pressures_list)):
                calypso_run_opt_path.append(
                    "%s.%03d" % (_calypso_run_opt_path, temp_idx)
                )
        # to different directory
        # calypso_run_opt_path = ['gen_struc_analy.000','gen_struc_analy.001','gen_struc_analy.002',]
        for temp_calypso_run_opt_path in calypso_run_opt_path:
            create_path(temp_calypso_run_opt_path)
            # run confs opt script
            run_opt_script = os.path.join(
                temp_calypso_run_opt_path, "calypso_run_opt.py"
            )
            shutil.copyfile(run_opt_file, run_opt_script)
            # check outcar script
            check_outcar_script = os.path.join(
                temp_calypso_run_opt_path, "check_outcar.py"
            )
            shutil.copyfile(check_outcar_file, check_outcar_script)

    for mm in models:
        model_name = os.path.basename(mm)
        if model_devi_engine != "calypso":
            os.symlink(mm, os.path.join(work_path, model_name))
        else:
            for temp_calypso_run_opt_path in calypso_run_opt_path:
                models_path = os.path.join(temp_calypso_run_opt_path, model_name)
                if not os.path.exists(models_path):
                    os.symlink(mm, models_path)

    with open(os.path.join(work_path, "cur_job.json"), "w") as outfile:
        json.dump(cur_job, outfile, indent=4)

    conf_path = os.path.join(work_path, "confs")
    create_path(conf_path)
    sys_counter = 0
    for ss in conf_systems:
        conf_counter = 0
        for cc in ss:
            if (model_devi_engine == "lammps") or (model_devi_engine == "dimer"):
                conf_name = make_model_devi_conf_name(
                    sys_idx[sys_counter], conf_counter
                )
                orig_poscar_name = conf_name + ".orig.poscar"
                poscar_name = conf_name + ".poscar"
                lmp_name = conf_name + ".lmp"
                if shuffle_poscar:
                    os.symlink(cc, os.path.join(conf_path, orig_poscar_name))
                    poscar_shuffle(
                        os.path.join(conf_path, orig_poscar_name),
                        os.path.join(conf_path, poscar_name),
                    )
                else:
                    os.symlink(cc, os.path.join(conf_path, poscar_name))
                if "sys_format" in jdata:
                    fmt = jdata["sys_format"]
                else:
                    fmt = "vasp/poscar"
                system = dpdata.System(
                    os.path.join(conf_path, poscar_name),
                    fmt=fmt,
                    type_map=jdata["type_map"],
                )
                if jdata.get("model_devi_nopbc", False):
                    system.remove_pbc()
                system.to_lammps_lmp(os.path.join(conf_path, lmp_name))
            elif model_devi_engine == "gromacs":
                pass
            elif model_devi_engine == "amber":
                # Jinzhe's specific Amber version
                conf_name = make_model_devi_conf_name(
                    sys_idx[sys_counter], conf_counter
                )
                rst7_name = conf_name + ".rst7"
                # link restart file
                os.symlink(cc, os.path.join(conf_path, rst7_name))
            conf_counter += 1
        sys_counter += 1

    input_mode = "native"
    if "calypso_input_path" in jdata:
        input_mode = "buffet"
    if "template" in cur_job:
        input_mode = "revise_template"
    use_plm = jdata.get("model_devi_plumed", False)
    use_plm_path = jdata.get("model_devi_plumed_path", False)
    if input_mode == "native":
        if (model_devi_engine == "lammps") or (model_devi_engine == "dimer"):
            _make_model_devi_native(iter_index, jdata, mdata, conf_systems)
        elif model_devi_engine == "gromacs":
            _make_model_devi_native_gromacs(iter_index, jdata, mdata, conf_systems)
        elif model_devi_engine == "amber":
            _make_model_devi_amber(iter_index, jdata, mdata, conf_systems)
        elif model_devi_engine == "calypso":
            _make_model_devi_native_calypso(
                iter_index, model_devi_jobs, calypso_run_opt_path
            )  # generate input.dat automatic in each iter
        else:
            raise RuntimeError("unknown model_devi engine", model_devi_engine)
    elif input_mode == "revise_template":
        _make_model_devi_revmat(iter_index, jdata, mdata, conf_systems)
    elif input_mode == "buffet":
        _make_model_devi_buffet(
            jdata, calypso_run_opt_path
        )  # generate confs according to the input.dat provided
    else:
        raise RuntimeError("unknown model_devi input mode", input_mode)
    # Copy user defined forward_files
    symlink_user_forward_files(mdata=mdata, task_type="model_devi", work_path=work_path)
    return True


def run_model_devi(iter_index, jdata, mdata):

    model_devi_engine = jdata.get("model_devi_engine", "lammps")
    if model_devi_engine != "calypso":
        run_md_model_devi(iter_index, jdata, mdata)
    else:
        run_calypso_model_devi(iter_index, jdata, mdata)






def post_model_devi(iter_index, jdata, mdata):
    pass






def run_md_model_devi(iter_index, jdata, mdata):
    logging.info("Running MD model deviation for iteration %d", iter_index)
    
    model_devi_exec = mdata["model_devi_command"]
    logging.info("Model deviation command: %s", model_devi_exec)

    model_devi_group_size = mdata["model_devi_group_size"]
    model_devi_resources = mdata["model_devi_resources"]
    logging.info("Model deviation group size: %d", model_devi_group_size)
    logging.info("Model deviation resources: %s", model_devi_resources)
    
    use_plm = jdata.get("model_devi_plumed", False)
    use_plm_path = jdata.get("model_devi_plumed_path", False)
    model_devi_merge_traj = jdata.get("model_devi_merge_traj", False)
    logging.info("Use PLUMED: %s", use_plm)
    logging.info("Use PLUMED path: %s", use_plm_path)
    logging.info("Merge trajectories: %s", model_devi_merge_traj)

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, model_devi_name)
    assert os.path.isdir(work_path), f"Work path {work_path} does not exist"
    logging.info("Work path: %s", work_path)

    all_task = glob.glob(os.path.join(work_path, "task.*"))
    all_task.sort()
    logging.info("Found %d tasks", len(all_task))
    
    fp = open(os.path.join(work_path, "cur_job.json"), "r")
    cur_job = json.load(fp)
    logging.info("Current job loaded from %s", os.path.join(work_path, "cur_job.json"))

    run_tasks_ = all_task
    run_tasks = [os.path.basename(ii) for ii in run_tasks_]
    logging.info("Running tasks: %s", run_tasks)









    
    # 记录在工作文件夹中搜索模型文件
    logging.info("Searching for model files in pattern 'graph*pb' under working folder %s...", work_path)

    # 使用 glob 搜索模型文件
    all_models = glob.glob(os.path.join(work_path, "graph*pb"))

    # 记录找到的模型文件数量和具体的文件路径列表  
    logging.info("Found %d model files: %s", len(all_models), all_models)

    # 提取模型文件的文件名列表
    model_names = [os.path.basename(ii) for ii in all_models]

    # 记录提取出的模型文件名列表
    logging.info("Extracted model file names: %s", model_names)

    # 打印最终的模型文件数量和文件名列表
    logging.info("Found %d models in total: %s", len(model_names), model_names)










    

    model_devi_engine = jdata.get("model_devi_engine", "lammps")
    logging.info("Model deviation engine: %s", model_devi_engine)
    
    if model_devi_engine == "lammps":
        command = (
            "{ if [ ! -f dpgen.restart.10000 ]; then %s -i input.lammps -v restart 0; else %s -i input.lammps -v restart 1; fi }"
            % (model_devi_exec, model_devi_exec)
        )
        command = "/bin/sh -c '%s'" % command
        commands = [command]
        logging.info("LAMMPS commands: %s", commands)

        forward_files = ["conf.lmp", "input.lammps"]
        backward_files = ["model_devi.out", "model_devi.log"]
        if model_devi_merge_traj:
            backward_files += ["all.lammpstrj"]
        else:
            forward_files += ["traj"]
            backward_files += ["traj"]
        logging.info("LAMMPS forward files: %s", forward_files)
        logging.info("LAMMPS backward files: %s", backward_files)

        if use_plm:
            forward_files += ["input.plumed"]
            backward_files += ["output.plumed", "COLVAR"]
            if use_plm_path:
                forward_files += ["plmpath.pdb"]
            logging.info("PLUMED enabled, updated forward files: %s", forward_files)
            logging.info("PLUMED enabled, updated backward files: %s", backward_files)
            
    elif model_devi_engine == "dimer":
        command = mdata["model_devi_command"] 
        commands = [command]
        logging.info("Dimer commands: %s", commands)

        forward_files = ["conf.lmp", "input.lammps"]
        backward_files = ["model_devi.out", "model_devi.log"]
        if model_devi_merge_traj:
            backward_files += ["all.lammpstrj"]
        else:
            forward_files += ["traj"]
            backward_files += ["traj"]
        logging.info("Dimer forward files: %s", forward_files)
        logging.info("Dimer backward files: %s", backward_files)
        
    elif model_devi_engine == "gromacs":
        gromacs_settings = jdata.get("gromacs_settings", {})
        mdp_filename = gromacs_settings.get("mdp_filename", "md.mdp")
        topol_filename = gromacs_settings.get("topol_filename", "processed.top")
        conf_filename = gromacs_settings.get("conf_filename", "conf.gro")
        index_filename = gromacs_settings.get("index_filename", "index.raw")
        type_filename = gromacs_settings.get("type_filename", "type.raw")
        ndx_filename = gromacs_settings.get("ndx_filename", "")
        ref_filename = gromacs_settings.get("ref_filename", "em.tpr")
        deffnm = gromacs_settings.get("deffnm", "deepmd")
        maxwarn = gromacs_settings.get("maxwarn", 1)
        traj_filename = gromacs_settings.get("traj_filename", "deepmd_traj.gro")
        grp_name = gromacs_settings.get("group_name", "Other")
        trj_freq = cur_job.get("trj_freq", 10)
        logging.info("GROMACS settings: %s", gromacs_settings)

        command = "%s grompp -f %s -p %s -c %s -o %s -maxwarn %d" % (
            model_devi_exec,
            mdp_filename,
            topol_filename,
            conf_filename,
            deffnm,
            maxwarn,
        )
        command += "&& %s mdrun -deffnm %s -cpi" % (model_devi_exec, deffnm)
        if ndx_filename:
            command += f'&& echo -e "{grp_name}\\n{grp_name}\\n" | {model_devi_exec} trjconv -s {ref_filename} -f {deffnm}.trr -n {ndx_filename} -o {traj_filename} -pbc mol -ur compact -center'
        else:
            command += (
                '&& echo -e "%s\\n%s\\n" | %s trjconv -s %s -f %s.trr -o %s -pbc mol -ur compact -center'
                % (
                    grp_name,
                    grp_name,
                    model_devi_exec,
                    ref_filename,
                    deffnm,
                    traj_filename,
                )
            )
        command += "&& if [ ! -d traj ]; then \n mkdir traj; fi\n"
        command += f"python -c \"import dpdata;system = dpdata.System('{traj_filename}', fmt='gromacs/gro'); [system.to_gromacs_gro('traj/%d.gromacstrj' % (i * {trj_freq}), frame_idx=i) for i in range(system.get_nframes())]; system.to_deepmd_npy('traj_deepmd')\""
        command += f"&& dp model-devi -m ../graph.000.pb ../graph.001.pb ../graph.002.pb ../graph.003.pb -s traj_deepmd -o model_devi.out -f {trj_freq}"
        commands = [command]
        logging.info("GROMACS commands: %s", commands)

        forward_files = [
            mdp_filename,
            topol_filename,
            conf_filename,
            index_filename,
            ref_filename,
            type_filename,
            "input.json",
            "job.json",
        ]
        if ndx_filename:
            forward_files.append(ndx_filename)
        backward_files = [
            "%s.tpr" % deffnm,
            "%s.log" % deffnm,
            traj_filename,
            "model_devi.out",
            "traj",
            "traj_deepmd",
        ]
        logging.info("GROMACS forward files: %s", forward_files)
        logging.info("GROMACS backward files: %s", backward_files)
        
    elif model_devi_engine == "amber":
        commands = [
            (
                "TASK=$(basename $(pwd)) && "
                "SYS1=${TASK:5:3} && "
                "SYS=$((10#$SYS1)) && "
            )
            + model_devi_exec
            + (
                " -O -p ../qmmm$SYS.parm7 -c init.rst7 -i ../init$SYS.mdin -o rc.mdout -r rc.rst7 -x rc.nc -inf rc.mdinfo -ref init.rst7"
            )
        ]
        logging.info("AMBER commands: %s", commands)
        
        forward_files = ["init.rst7", "TEMPLATE.disang"]
        backward_files = ["rc.mdout", "rc.nc", "rc.rst7", "TEMPLATE.dumpave"]
        model_names.extend(["qmmm*.parm7", "init*.mdin"])
        logging.info("AMBER forward files: %s", forward_files)
        logging.info("AMBER backward files: %s", backward_files)
        logging.info("AMBER model names: %s", model_names)

    cwd = os.getcwd()
    logging.info("Current working directory: %s", cwd)

    user_forward_files = mdata.get("model_devi" + "_user_forward_files", [])
    forward_files += [os.path.basename(file) for file in user_forward_files]
    backward_files += mdata.get("model_devi" + "_user_backward_files", [])
    logging.info("User-defined forward files: %s", user_forward_files)
    logging.info("Updated forward files: %s", forward_files)
    logging.info("Updated backward files: %s", backward_files)
    
    api_version = mdata.get("api_version", "1.0")
    logging.info("API version: %s", api_version)

    if len(run_tasks) == 0:
        logging.error("No tasks to run for model deviation")
        raise RuntimeError(
            "run_tasks for model_devi should not be empty! Please check your files."
        )

    if Version(api_version) < Version("1.0"):
        logging.error("API version %s is no longer supported", api_version)
        raise RuntimeError(
            "API version %s has been removed. Please upgrade to 1.0." % api_version
        )

    elif Version(api_version) >= Version("1.0"):
        submission = make_submission(
            mdata["model_devi_machine"],
            mdata["model_devi_resources"],
            commands=commands,
            work_path=work_path,
            run_tasks=run_tasks,
            group_size=model_devi_group_size,
            forward_common_files=model_names,
            forward_files=forward_files,
            backward_files=backward_files,
            outlog="model_devi.log",
            errlog="model_devi.log",
        )
        logging.info("Submission created for model deviation")
        submission.run_submission()
        logging.info("Model deviation submission completed")







def make_fp(iter_index, jdata, mdata):
    """Generate first-principles data.

    Args:
        iter_index (int): Current iteration index.
        jdata (dict): Job configuration dictionary.
        mdata (dict): Machine configuration dictionary.
    """
    fp_style = jdata['fp_style']
    if fp_style == 'vasp':
        make_fp_vasp(iter_index, jdata)
    elif fp_style == 'pwscf':
        make_fp_pwscf(iter_index, jdata)
    elif fp_style == 'siesta':
        make_fp_siesta(iter_index, jdata)
    elif fp_style == 'gaussian':
        make_fp_gaussian(iter_index, jdata)
    elif fp_style == 'cp2k':
        make_fp_cp2k(iter_index, jdata)
    else:
        raise RuntimeError('unsupported fp style')










def run_fp(iter_index, jdata, mdata):
    fp_style = jdata["fp_style"]
    fp_pp_files = jdata.get("fp_pp_files", [])

    if fp_style == "vasp":
        forward_files = ["POSCAR", "INCAR", "POTCAR", "KPOINTS"]
        # forward_files = ["input/POSCAR", "input/INCAR", "input/POTCAR", "input/KPOINTS"]

        backward_files = ["fp.log", "OUTCAR", "vasprun.xml"]
        # Move cvasp interface to jdata
        if ("cvasp" in jdata) and (jdata["cvasp"] == True):
            mdata["fp_resources"]["cvasp"] = True
        if ("cvasp" in mdata["fp_resources"]) and (
            mdata["fp_resources"]["cvasp"] == True
        ):
            dlog.info("cvasp is on !")
            forward_files.append("cvasp.py")
            forward_common_files = []
        else:
            forward_common_files = []
        run_fp_inner(
            iter_index,
            jdata,
            mdata,
            forward_files,
            backward_files,
            _vasp_check_fin,
            forward_common_files=forward_common_files,
        )
    elif fp_style == "pwscf":
        forward_files = ["input"] + fp_pp_files
        backward_files = ["output"]
        run_fp_inner(
            iter_index,
            jdata,
            mdata,
            forward_files,
            backward_files,
            _qe_check_fin,
            log_file="output",
        )
    elif fp_style == "abacus":
        fp_params = {}
        if "user_fp_params" in jdata.keys():
            fp_params = jdata["user_fp_params"]
        elif "fp_incar" in jdata.keys():
            fp_input_path = jdata["fp_incar"]
            assert os.path.exists(fp_input_path)
            fp_input_path = os.path.abspath(fp_input_path)
            fp_params = get_abacus_input_parameters(fp_input_path)
        forward_files = ["INPUT", "STRU"]
        if "kspacing" not in fp_params.keys():
            forward_files = ["INPUT", "STRU", "KPT"]
        forward_files += fp_pp_files
        if "fp_orb_files" in jdata:
            forward_files += jdata["fp_orb_files"]
        if "fp_dpks_descriptor" in jdata:
            forward_files.append(jdata["fp_dpks_descriptor"])
        if "user_fp_params" in jdata:
            if "deepks_model" in jdata["user_fp_params"]:
                forward_files.append(jdata["user_fp_params"]["deepks_model"])
        backward_files = ["output", "OUT.ABACUS"]
        run_fp_inner(
            iter_index,
            jdata,
            mdata,
            forward_files,
            backward_files,
            _abacus_scf_check_fin,
            log_file="output",
        )
    elif fp_style == "siesta":
        forward_files = ["input"] + fp_pp_files
        backward_files = ["output"]
        run_fp_inner(
            iter_index,
            jdata,
            mdata,
            forward_files,
            backward_files,
            _siesta_check_fin,
            log_file="output",
        )
    elif fp_style == "gaussian":
        forward_files = ["input"]
        backward_files = ["output"]
        run_fp_inner(
            iter_index,
            jdata,
            mdata,
            forward_files,
            backward_files,
            _gaussian_check_fin,
            log_file="output",
        )
    elif fp_style == "cp2k":
        forward_files = ["input.inp", "coord.xyz"]
        backward_files = ["output"]
        run_fp_inner(
            iter_index,
            jdata,
            mdata,
            forward_files,
            backward_files,
            _cp2k_check_fin,
            log_file="output",
        )
    elif fp_style == "pwmat":
        forward_files = ["atom.config", "etot.input"] + fp_pp_files
        backward_files = ["REPORT", "OUT.MLMD", "output"]
        run_fp_inner(
            iter_index,
            jdata,
            mdata,
            forward_files,
            backward_files,
            _pwmat_check_fin,
            log_file="output",
        )
    elif fp_style == "amber/diff":
        forward_files = ["rc.nc"]
        backward_files = [
            "low_level.mdfrc",
            "low_level.mdout",
            "high_level.mdfrc",
            "high_level.mdout",
            "output",
            "dataset",
        ]
        forward_common_files = [
            "low_level*.mdin",
            "high_level*.mdin",
            "qmmm*.parm7",
            "qm_region",
            "init*.rst7",
        ]
        run_fp_inner(
            iter_index,
            jdata,
            mdata,
            forward_files,
            backward_files,
            None,
            log_file="output",
            forward_common_files=forward_common_files,
        )
    else:
        raise RuntimeError("unsupported fp style")








def post_fp(iter_index, jdata):
    """Post-process first-principles calculations.

    Args:
        iter_index (int): Current iteration index.
        jdata (dict): Job configuration dictionary.
    """
    fp_style = jdata['fp_style']
    if fp_style == 'vasp':
        post_fp_vasp(iter_index, jdata)
    elif fp_style == 'pwscf':
        post_fp_pwscf(iter_index, jdata)
    elif fp_style == 'siesta':
        post_fp_siesta(iter_index, jdata)
    elif fp_style == 'gaussian':
        post_fp_gaussian(iter_index, jdata)
    elif fp_style == 'cp2k':
        post_fp_cp2k(iter_index, jdata)
    else:
        raise RuntimeError('unsupported fp style')



def run_fp_inner(
    iter_index,
    jdata,
    mdata,
    forward_files,
    backward_files,
    check_fin,
    log_file="fp.log",
    forward_common_files=[],
):
    fp_command = mdata["fp_command"]
    fp_group_size = mdata["fp_group_size"]
    fp_resources = mdata["fp_resources"]
    mark_failure = fp_resources.get("mark_failure", False)

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)

    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return

    fp_style = jdata["fp_style"]
    if fp_style == "amber/diff":
        # firstly get sys_idx
        fp_command = (
            (
                "TASK=$(basename $(pwd)) && "
                "SYS1=${TASK:5:3} && "
                "SYS=$((10#$SYS1)) && "
                'QM_REGION=$(awk "NR==$SYS+1" ../qm_region) &&'
            )
            + fp_command
            + (
                " -O -p ../qmmm$SYS.parm7 -c ../init$SYS.rst7 -i ../low_level$SYS.mdin -o low_level.mdout -r low_level.rst7 "
                "-x low_level.nc -y rc.nc -frc low_level.mdfrc -inf low_level.mdinfo && "
            )
            + fp_command
            + (
                " -O -p ../qmmm$SYS.parm7 -c ../init$SYS.rst7 -i ../high_level$SYS.mdin -o high_level.mdout -r high_level.rst7 "
                "-x high_level.nc -y rc.nc -frc high_level.mdfrc -inf high_level.mdinfo && "
            )
            + (
                'dpamber corr --cutoff %f --parm7_file ../qmmm$SYS.parm7 --nc rc.nc --hl high_level --ll low_level --qm_region "$QM_REGION"'
            )
            % (jdata["cutoff"],)
        )

    fp_run_tasks = fp_tasks
    # for ii in fp_tasks :
    #     if not check_fin(ii) :
    #         fp_run_tasks.append(ii)
    run_tasks = [os.path.basename(ii) for ii in fp_run_tasks]

    user_forward_files = mdata.get("fp" + "_user_forward_files", [])
    forward_files += [os.path.basename(file) for file in user_forward_files]
    backward_files += mdata.get("fp" + "_user_backward_files", [])

    api_version = mdata.get("api_version", "1.0")
    if Version(api_version) < Version("1.0"):
        raise RuntimeError(
            "API version %s has been removed. Please upgrade to 1.0." % api_version
        )

    elif Version(api_version) >= Version("1.0"):
        submission = make_submission(
            mdata["fp_machine"],
            mdata["fp_resources"],
            commands=[fp_command],
            work_path=work_path,
            run_tasks=run_tasks,
            group_size=fp_group_size,
            forward_common_files=forward_common_files,
            forward_files=forward_files,
            backward_files=backward_files,
            outlog=log_file,
            errlog=log_file,
        )
        submission.run_submission()







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







def _check_empty_iter(iter_index, max_v=0):
    fp_path = os.path.join(make_iter_name(iter_index), fp_name)
    # check the number of collected data
    sys_data = glob.glob(os.path.join(fp_path, "data.*"))
    empty_sys = []
    for ii in sys_data:
        nframe = 0
        sys_paths = expand_sys_str(ii)
        for single_sys in sys_paths:
            sys = dpdata.LabeledSystem(os.path.join(single_sys), fmt="deepmd/npy")
            nframe += len(sys)
        empty_sys.append(nframe < max_v)
    return all(empty_sys)




def copy_model(numb_model, prv_iter_index, cur_iter_index):
    cwd = os.getcwd()
    prv_train_path = os.path.join(make_iter_name(prv_iter_index), train_name)
    cur_train_path = os.path.join(make_iter_name(cur_iter_index), train_name)
    prv_train_path = os.path.abspath(prv_train_path)
    cur_train_path = os.path.abspath(cur_train_path)
    create_path(cur_train_path)
    for ii in range(numb_model):
        prv_train_task = os.path.join(prv_train_path, train_task_fmt % ii)
        os.chdir(cur_train_path)
        os.symlink(os.path.relpath(prv_train_task), train_task_fmt % ii)
        os.symlink(
            os.path.join(train_task_fmt % ii, "frozen_model.pb"), "graph.%03d.pb" % ii
        )
        os.chdir(cwd)
    with open(os.path.join(cur_train_path, "copied"), "w") as fp:
        None



# region 删除
def run_fp_ORGINAL(iter_index, jdata, mdata):
    """Run first-principles calculations.

    Args:
        iter_index (int): Current iteration index.
        jdata (dict): Job configuration dictionary.
        mdata (dict): Machine configuration dictionary.
    """
    fp_style = jdata['fp_style']
    if fp_style == 'vasp':
        forward_files = ['POSCAR', 'INCAR', 'POTCAR', 'KPOINTS']
        backward_files = ['fp.log', 'OUTCAR', 'vasprun.xml']
        run_fp_inner(iter_index, jdata, mdata, forward_files, backward_files, _vasp_check_fin)
    elif fp_style == 'pwscf':
        forward_files = ['input'] + jdata.get('fp_pp_files', [])
        backward_files = ['output']
        run_fp_inner(iter_index, jdata, mdata, forward_files, backward_files, _qe_check_fin, log_file='output')
    elif fp_style == 'siesta':
        forward_files = ['input'] + jdata.get('fp_pp_files', [])
        backward_files = ['output']
        run_fp_inner(iter_index, jdata, mdata, forward_files, backward_files, _siesta_check_fin, log_file='output')
    elif fp_style == 'gaussian':
        forward_files = ['input']
        backward_files = ['output']
        run_fp_inner(iter_index, jdata, mdata, forward_files, backward_files, _gaussian_check_fin, log_file='output')
    elif fp_style == 'cp2k':
        forward_files = ['input.inp', 'coord.xyz']
        backward_files = ['output']
        run_fp_inner(iter_index, jdata, mdata, forward_files, backward_files, _cp2k_check_fin, log_file='output')
    else:
        raise RuntimeError('unsupported fp style')





def run_fp_ORGINAL_2(iter_index, jdata, mdata):
    fp_style = jdata['fp_style']
    
    if fp_style == 'vasp':
        forward_files = ['POSCAR', 'INCAR', 'POTCAR', 'KPOINTS']
        backward_files = ['fp.log', 'OUTCAR', 'vasprun.xml']
        fp_check_fin = _vasp_check_fin
    elif fp_style == 'pwscf':
        forward_files = ['input'] + jdata.get('fp_pp_files', [])
        backward_files = ['output']
        fp_check_fin = _qe_check_fin
    elif fp_style == 'siesta':
        forward_files = ['input'] + jdata.get('fp_pp_files', [])
        backward_files = ['output']
        fp_check_fin = _siesta_check_fin
    elif fp_style == 'gaussian':
        forward_files = ['input']
        backward_files = ['output']
        fp_check_fin = _gaussian_check_fin
    elif fp_style == 'cp2k':
        forward_files = ['input.inp', 'coord.xyz']
        backward_files = ['output']
        fp_check_fin = _cp2k_check_fin
    else:
        raise RuntimeError('unsupported fp style')

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, 'task.*'))
    
    for ii in fp_tasks:
        # check if input files exist
        for jj in forward_files:
            if not os.path.exists(os.path.join(ii, jj)):
                logging.warning(f'Input file {jj} not found for task {ii}, skipping')
                continue
        
        # submit task
        logging.info(f'Submitting task {ii}')
        task_name = os.path.basename(ii)
        run_fp_inner(iter_index, jdata, mdata, forward_files, backward_files, fp_check_fin, log_file=task_name+'.log')
        
        # check if task finished successfully
        if not fp_check_fin(ii):
            logging.error(f'Task {ii} not finished successfully')
            continue
        
        # download output files
        logging.info(f'Downloading output files for task {ii}')
        for jj in backward_files:
            try:
                # download file
                logging.info(f'Downloading file {jj}')
            except Exception as e:
                logging.warning(f'Failed to download file {jj}: {e}')
        
    # check if all tasks finished successfully
    if not all(fp_check_fin(ii) for ii in fp_tasks):
        logging.error('Some tasks not finished successfully')



def run_md_model_devi_legacy(iter_index, jdata, mdata):
    """
    Run the model deviation task for molecular dynamics simulations.

    Args:
        iter_index (int): The current iteration index.
        jdata (dict): The job configuration data.
        mdata (dict): The machine configuration data.
    """
    try:
        # Get the model deviation execution command from the machine configuration
        model_devi_exec = mdata["model_devi_command"]
    except KeyError as e:
        logging.error(f"Error retrieving model_devi_command from mdata: {str(e)}")
        raise

    try:
        # Get the model deviation group size and resource requirements from the machine configuration
        model_devi_group_size = mdata["model_devi_group_size"]
        model_devi_resources = mdata["model_devi_resources"]
    except KeyError as e:
        logging.error(f"Error retrieving model_devi_group_size or model_devi_resources from mdata: {str(e)}")
        raise

    try:
        # Get the PLUMED-related settings from the job configuration
        use_plm = jdata.get("model_devi_plumed", False)
        use_plm_path = jdata.get("model_devi_plumed_path", False)
        model_devi_merge_traj = jdata.get("model_devi_merge_traj", False)
    except Exception as e:
        logging.error(f"Error retrieving PLUMED-related settings from jdata: {str(e)}")
        raise

    # Construct the iteration name and work path
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, MODEL_DEVI_NAME)
    logging.info(f"Model deviation work path: {work_path}")

    # Check if the work path exists
    if not os.path.isdir(work_path):
        raise FileNotFoundError(f"Model deviation work path not found: {work_path}")

    try:
        # Get all the task directories
        all_tasks = glob.glob(os.path.join(work_path, "task.*"))
        all_tasks.sort()
    except Exception as e:
        logging.error(f"Error retrieving task directories: {str(e)}")
        raise

    try:
        # Load the current job configuration
        with open(os.path.join(work_path, "cur_job.json"), "r") as fp:
            cur_job = json.load(fp)
    except Exception as e:
        logging.error(f"Error loading current job configuration: {str(e)}")
        raise

    # Set the run tasks to all tasks
    run_tasks = [os.path.basename(task) for task in all_tasks]

    try:
        # Get all the model files
        all_models = glob.glob(os.path.join(work_path, "graph*pb"))
        model_names = [os.path.basename(model) for model in all_models]
    except Exception as e:
        logging.error(f"Error retrieving model files: {str(e)}")
        raise

    try:
        # Get the model deviation engine from the job configuration (default: "lammps")
        model_devi_engine = jdata.get("model_devi_engine", "lammps")
    except Exception as e:
        logging.error(f"Error retrieving model_devi_engine from jdata: {str(e)}")
        raise

    # Set the commands, forward files, and backward files based on the model deviation engine
    if model_devi_engine == "lammps":
        try:
            command = (
                "{ if [ ! -f dpgen.restart.10000 ]; then %s -i input.lammps -v restart 0; else %s -i input.lammps -v restart 1; fi }"
                % (model_devi_exec, model_devi_exec)
            )
            command = "/bin/sh -c '%s'" % command
            commands = [command]
        except Exception as e:
            logging.error(f"Error constructing lammps command: {str(e)}")
            raise

        try:
            forward_files = ["conf.lmp", "input.lammps"]
            backward_files = ["model_devi.out", "model_devi.log"]
            if model_devi_merge_traj:
                backward_files += ["all.lammpstrj"]
            else:
                forward_files += ["traj"]
                backward_files += ["traj"]

            if use_plm:
                forward_files += ["input.plumed"]
                backward_files += ["output.plumed", "COLVAR"]
                if use_plm_path:
                    forward_files += ["plmpath.pdb"]
        except Exception as e:
            logging.error(f"Error constructing forward and backward files for lammps: {str(e)}")
            raise

    elif model_devi_engine == "dimer":
        try:
            command = mdata["model_devi_command"]
            commands = [command]
        except KeyError as e:
            logging.error(f"Error retrieving model_devi_command from mdata for dimer: {str(e)}")
            raise

        try:
            forward_files = ["conf.lmp", "input.lammps"]
            backward_files = ["model_devi.out", "model_devi.log"]
            if model_devi_merge_traj:
                backward_files += ["all.lammpstrj"]
            else:
                forward_files += ["traj"]
                backward_files += ["traj"]
        except Exception as e:
            logging.error(f"Error constructing forward and backward files for dimer: {str(e)}")
            raise

    elif model_devi_engine == "gromacs":
        try:
            gromacs_settings = jdata.get("gromacs_settings", {})
            mdp_filename = gromacs_settings.get("mdp_filename", "md.mdp")
            topol_filename = gromacs_settings.get("topol_filename", "processed.top")
            conf_filename = gromacs_settings.get("conf_filename", "conf.gro")
            index_filename = gromacs_settings.get("index_filename", "index.raw")
            type_filename = gromacs_settings.get("type_filename", "type.raw")
            ndx_filename = gromacs_settings.get("ndx_filename", "")
            ref_filename = gromacs_settings.get("ref_filename", "em.tpr")
            deffnm = gromacs_settings.get("deffnm", "deepmd")
            maxwarn = gromacs_settings.get("maxwarn", 1)
            traj_filename = gromacs_settings.get("traj_filename", "deepmd_traj.gro")
            grp_name = gromacs_settings.get("group_name", "Other")
            trj_freq = cur_job.get("trj_freq", 10)
        except Exception as e:
            logging.error(f"Error retrieving gromacs settings: {str(e)}")
            raise

        try:
            command = "%s grompp -f %s -p %s -c %s -o %s -maxwarn %d" % (
                model_devi_exec,
                mdp_filename,
                topol_filename,
                conf_filename,
                deffnm,
                maxwarn,
            )
            command += "&& %s mdrun -deffnm %s -cpi" % (model_devi_exec, deffnm)
            if ndx_filename:
                command += f'&& echo -e "{grp_name}\\n{grp_name}\\n" | {model_devi_exec} trjconv -s {ref_filename} -f {deffnm}.trr -n {ndx_filename} -o {traj_filename} -pbc mol -ur compact -center'
            else:
                command += (
                    '&& echo -e "%s\\n%s\\n" | %s trjconv -s %s -f %s.trr -o %s -pbc mol -ur compact -center'
                    % (
                        grp_name,
                        grp_name,
                        model_devi_exec,
                        ref_filename,
                        deffnm,
                        traj_filename,
                    )
                )
            command += "&& if [ ! -d traj ]; then \n mkdir traj; fi\n"
            command += f"python -c \"import dpdata;system = dpdata.System('{traj_filename}', fmt='gromacs/gro'); [system.to_gromacs_gro('traj/%d.gromacstrj' % (i * {trj_freq}), frame_idx=i) for i in range(system.get_nframes())]; system.to_deepmd_npy('traj_deepmd')\""
            command += f"&& dp model-devi -m ../graph.000.pb ../graph.001.pb ../graph.002.pb ../graph.003.pb -s traj_deepmd -o model_devi.out -f {trj_freq}"
            commands = [command]
        except Exception as e:
            logging.error(f"Error constructing gromacs command: {str(e)}")
            raise

        try:
            forward_files = [
                mdp_filename,
                topol_filename,
                conf_filename,
                index_filename,
                ref_filename,
                type_filename,
                "input.json",
                "job.json",
            ]
            if ndx_filename:
                forward_files.append(ndx_filename)
            backward_files = [
                "%s.tpr" % deffnm,
                "%s.log" % deffnm,
                traj_filename,
                "model_devi.out",
                "traj",
                "traj_deepmd",
            ]
        except Exception as e:
            logging.error(f"Error constructing forward and backward files for gromacs: {str(e)}")
            raise

    elif model_devi_engine == "amber":
        try:
            commands = [
                (
                    "TASK=$(basename $(pwd)) && "
                    "SYS1=${TASK:5:3} && "
                    "SYS=$((10#$SYS1)) && "
                )
                + model_devi_exec
                + (
                    " -O -p ../qmmm$SYS.parm7 -c init.rst7 -i ../init$SYS.mdin -o rc.mdout -r rc.rst7 -x rc.nc -inf rc.mdinfo -ref init.rst7"
                )
            ]
            forward_files = ["init.rst7", "TEMPLATE.disang"]
            backward_files = ["rc.mdout", "rc.nc", "rc.rst7", "TEMPLATE.dumpave"]
            model_names.extend(["qmmm*.parm7", "init*.mdin"])
        except Exception as e:
            logging.error(f"Error constructing commands and files for amber: {str(e)}")
            raise

    else:
        raise ValueError(f"Unknown model deviation engine: {model_devi_engine}")

    try:
        # Get the current working directory
        cwd = os.getcwd()
    except Exception as e:
        logging.error(f"Error retrieving current working directory: {str(e)}")
        raise

    try:
        # Get the user-defined forward and backward files from the machine configuration
        user_forward_files = mdata.get("model_devi" + "_user_forward_files", [])
        forward_files += [os.path.basename(file) for file in user_forward_files]
        backward_files += mdata.get("model_devi" + "_user_backward_files", [])
    except Exception as e:
        logging.error(f"Error retrieving user-defined forward and backward files: {str(e)}")
        raise

    try:
        # Get the API version from the machine configuration (default: "1.0")
        api_version = mdata.get("api_version", "1.0")
    except Exception as e:
        logging.error(f"Error retrieving API version from mdata: {str(e)}")
        raise

    # Check if there are any run tasks
    if not run_tasks:
        raise RuntimeError("No run tasks found for model deviation. Please check your files.")

    try:
        # Check the API version
        if Version(api_version) < Version("1.0"):
            raise RuntimeError(f"API version {api_version} has been removed. Please upgrade to 1.0 or higher.")
    except Exception as e:
        logging.error(f"Error checking API version: {str(e)}")
        raise

    try:
        # Create the submission object
        submission = make_submission(
            mdata["model_devi_machine"],
            mdata["model_devi_resources"],
            commands=commands,
            work_path=work_path,
            run_tasks=run_tasks,
            group_size=model_devi_group_size,
            forward_common_files=model_names,
            forward_files=forward_files,
            backward_files=backward_files,
            outlog="model_devi.log",
            errlog="model_devi.log",
        )
    except Exception as e:
        logging.error(f"Error creating submission object: {str(e)}")
        raise

    try:
        # Run the submission
        submission.run_submission()
    except Exception as e:
        # Log the error and raise the exception
        logging.error(f"Error occurred during model deviation submission: {str(e)}")
        raise



def run_md_model_devi_ORGINAL(iter_index, jdata, mdata):

    # rmdlog.info("This module has been run !")
    model_devi_exec = mdata["model_devi_command"]

    model_devi_group_size = mdata["model_devi_group_size"]
    model_devi_resources = mdata["model_devi_resources"]
    use_plm = jdata.get("model_devi_plumed", False)
    use_plm_path = jdata.get("model_devi_plumed_path", False)
    model_devi_merge_traj = jdata.get("model_devi_merge_traj", False)

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, model_devi_name)
    assert os.path.isdir(work_path)

    all_task = glob.glob(os.path.join(work_path, "task.*"))
    all_task.sort()
    fp = open(os.path.join(work_path, "cur_job.json"), "r")
    cur_job = json.load(fp)

    run_tasks_ = all_task
    # for ii in all_task:
    #     fres = os.path.join(ii, 'model_devi.out')
    #     if os.path.isfile(fres) :
    #         nlines = np.loadtxt(fres).shape[0]
    #         if nframes != nlines :
    #             run_tasks_.append(ii)
    #     else :
    #         run_tasks_.append(ii)

    run_tasks = [os.path.basename(ii) for ii in run_tasks_]
    # dlog.info("all_task is ", all_task)
    # dlog.info("run_tasks in run_model_deviation",run_tasks_)
    all_models = glob.glob(os.path.join(work_path, "graph*pb"))
    model_names = [os.path.basename(ii) for ii in all_models]

    model_devi_engine = jdata.get("model_devi_engine", "lammps")
    if model_devi_engine == "lammps":
        command = (
            "{ if [ ! -f dpgen.restart.10000 ]; then %s -i input.lammps -v restart 0; else %s -i input.lammps -v restart 1; fi }"
            % (model_devi_exec, model_devi_exec)
        )
        command = "/bin/sh -c '%s'" % command
        commands = [command]

        forward_files = ["conf.lmp", "input.lammps"]
        backward_files = ["model_devi.out", "model_devi.log"]
        if model_devi_merge_traj:
            backward_files += ["all.lammpstrj"]
        else:
            forward_files += ["traj"]
            backward_files += ["traj"]

        if use_plm:
            forward_files += ["input.plumed"]
            # backward_files += ['output.plumed']
            backward_files += ["output.plumed", "COLVAR"]
            if use_plm_path:
                forward_files += ["plmpath.pdb"]
    elif model_devi_engine == "dimer":
       # command = (
        #    "{ if [ ! -f dpgen.restart.10000 ]; then %s -i input.lammps -v restart 0; else %s -i input.lammps -v restart 1; fi }"
        #    % (model_devi_exec, model_devi_exec)
        #)
        command = mdata["model_devi_command"] 
        commands = [command]

        forward_files = ["conf.lmp", "input.lammps"]
        backward_files = ["model_devi.out", "model_devi.log"]
        if model_devi_merge_traj:
            backward_files += ["all.lammpstrj"]
        else:
            forward_files += ["traj"]
            backward_files += ["traj"]
    elif model_devi_engine == "gromacs":

        gromacs_settings = jdata.get("gromacs_settings", {})
        mdp_filename = gromacs_settings.get("mdp_filename", "md.mdp")
        topol_filename = gromacs_settings.get("topol_filename", "processed.top")
        conf_filename = gromacs_settings.get("conf_filename", "conf.gro")
        index_filename = gromacs_settings.get("index_filename", "index.raw")
        type_filename = gromacs_settings.get("type_filename", "type.raw")
        ndx_filename = gromacs_settings.get("ndx_filename", "")
        # Initial reference to process pbc condition.
        # Default is em.tpr
        ref_filename = gromacs_settings.get("ref_filename", "em.tpr")
        deffnm = gromacs_settings.get("deffnm", "deepmd")
        maxwarn = gromacs_settings.get("maxwarn", 1)
        traj_filename = gromacs_settings.get("traj_filename", "deepmd_traj.gro")
        grp_name = gromacs_settings.get("group_name", "Other")
        trj_freq = cur_job.get("trj_freq", 10)

        command = "%s grompp -f %s -p %s -c %s -o %s -maxwarn %d" % (
            model_devi_exec,
            mdp_filename,
            topol_filename,
            conf_filename,
            deffnm,
            maxwarn,
        )
        command += "&& %s mdrun -deffnm %s -cpi" % (model_devi_exec, deffnm)
        if ndx_filename:
            command += f'&& echo -e "{grp_name}\\n{grp_name}\\n" | {model_devi_exec} trjconv -s {ref_filename} -f {deffnm}.trr -n {ndx_filename} -o {traj_filename} -pbc mol -ur compact -center'
        else:
            command += (
                '&& echo -e "%s\\n%s\\n" | %s trjconv -s %s -f %s.trr -o %s -pbc mol -ur compact -center'
                % (
                    grp_name,
                    grp_name,
                    model_devi_exec,
                    ref_filename,
                    deffnm,
                    traj_filename,
                )
            )
        command += "&& if [ ! -d traj ]; then \n mkdir traj; fi\n"
        command += f"python -c \"import dpdata;system = dpdata.System('{traj_filename}', fmt='gromacs/gro'); [system.to_gromacs_gro('traj/%d.gromacstrj' % (i * {trj_freq}), frame_idx=i) for i in range(system.get_nframes())]; system.to_deepmd_npy('traj_deepmd')\""
        command += f"&& dp model-devi -m ../graph.000.pb ../graph.001.pb ../graph.002.pb ../graph.003.pb -s traj_deepmd -o model_devi.out -f {trj_freq}"
        commands = [command]

        forward_files = [
            mdp_filename,
            topol_filename,
            conf_filename,
            index_filename,
            ref_filename,
            type_filename,
            "input.json",
            "job.json",
        ]
        if ndx_filename:
            forward_files.append(ndx_filename)
        backward_files = [
            "%s.tpr" % deffnm,
            "%s.log" % deffnm,
            traj_filename,
            "model_devi.out",
            "traj",
            "traj_deepmd",
        ]
    elif model_devi_engine == "amber":
        commands = [
            (
                "TASK=$(basename $(pwd)) && "
                "SYS1=${TASK:5:3} && "
                "SYS=$((10#$SYS1)) && "
            )
            + model_devi_exec
            + (
                " -O -p ../qmmm$SYS.parm7 -c init.rst7 -i ../init$SYS.mdin -o rc.mdout -r rc.rst7 -x rc.nc -inf rc.mdinfo -ref init.rst7"
            )
        ]
        forward_files = ["init.rst7", "TEMPLATE.disang"]
        backward_files = ["rc.mdout", "rc.nc", "rc.rst7", "TEMPLATE.dumpave"]
        model_names.extend(["qmmm*.parm7", "init*.mdin"])

    cwd = os.getcwd()

    user_forward_files = mdata.get("model_devi" + "_user_forward_files", [])
    forward_files += [os.path.basename(file) for file in user_forward_files]
    backward_files += mdata.get("model_devi" + "_user_backward_files", [])
    api_version = mdata.get("api_version", "1.0")
    if len(run_tasks) == 0:
        raise RuntimeError(
            "run_tasks for model_devi should not be empty! Please check your files."
        )
    if Version(api_version) < Version("1.0"):
        raise RuntimeError(
            "API version %s has been removed. Please upgrade to 1.0." % api_version
        )

    elif Version(api_version) >= Version("1.0"):
        submission = make_submission(
            mdata["model_devi_machine"],
            mdata["model_devi_resources"],
            commands=commands,
            work_path=work_path,
            run_tasks=run_tasks,
            group_size=model_devi_group_size,
            forward_common_files=model_names,
            forward_files=forward_files,
            backward_files=backward_files,
            outlog="model_devi.log",
            errlog="model_devi.log",
        )
        submission.run_submission()

# endregion

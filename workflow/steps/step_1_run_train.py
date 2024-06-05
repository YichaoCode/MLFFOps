# workflow/steps/step_1_run_train.py
# Created by Yichao


# region
import os
import glob
import logging
from packaging.version import Version 

from packaging.version import Version


from dpgen.generator.lib.utils import make_iter_name
from dpgen.dispatcher.Dispatcher import make_submission

from config.config import *
from dpgen.util import expand_sys_str
# endregion


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

        if training_reuse_iter is not None and iter_index >= training_re.use_iter:
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

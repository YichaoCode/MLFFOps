# run_iter.py
# created by Yichao

import os
import json
import logging
import warnings

from workflow.steps.step_0_make_train import make_train
from workflow.steps.step_1_run_train import run_train
from workflow.steps.step_2_post_train import post_train
from workflow.steps.step_3_make_model_devi import make_model_devi
from workflow.steps.step_4_run_model_devi import run_model_devi
from workflow.steps.step_5_post_model_devi import post_model_devi
from workflow.steps.step_6_make_fp import make_fp
from workflow.steps.step_7_run_fp import run_fp
from workflow.steps.step_8_post_fp import post_fp

from dpgen.generator.lib.utils import (
    copy_file_list,
    create_path,
    log_iter,
    log_task,
    make_iter_name,
    record_iter,
    replace,
    symlink_user_forward_files,
)

from utils.utils import (
    update_mass_map
)

from utils.parse_utils import parse_cur_job
from arginfo import run_jdata_arginfo
from dpgen.util import normalize, sepline
from dpgen.remote.decide_machine import convert_mdata
from dpgen import ROOT_PATH, SHORT_CMD, dlog
from config import *

def run_iter(param_file, machine_file, restart_task=None):
    logger = logging.getLogger(__name__)

    try:
        import ruamel
        from monty.serialization import dumpfn, loadfn

        warnings.simplefilter("ignore", ruamel.yaml.error.MantissaNoDotYAML1_1Warning)

        logger.info(f"Loading parameter file: {param_file}")
        jdata = loadfn(param_file)
        logger.info(f"Parameter data loaded successfully")

        logger.info(f"Loading machine file: {machine_file}")
        mdata = loadfn(machine_file)
        logger.info(f"Machine data loaded successfully")

    except Exception as e:
        logger.warning(f"Error loading data from ruamel/monty serializers: {e}")
        logger.warning("Falling back to JSON loading")

        logger.info(f"Loading parameter file: {param_file}")
        with open(param_file, "r") as fp:
            jdata = json.load(fp)
        logger.info(f"Parameter data loaded successfully")

        logger.info(f"Loading machine file: {machine_file}")
        with open(machine_file, "r") as fp:
            mdata = json.load(fp)
        logger.info(f"Machine data loaded successfully")

    logger.info("Getting argument info for jdata")
    jdata_arginfo = run_jdata_arginfo()

    logger.info("Normalizing jdata based on argument info")
    jdata = normalize(jdata_arginfo, jdata, strict_check=False)

    logger.info("Updating mass map in jdata")
    update_mass_map(jdata)

    if jdata.get("pretty_print", False):
        logger.info("Pretty printing is enabled")

        fparam = f"{SHORT_CMD}_{param_file.split('.')[0]}.{jdata.get('pretty_format', 'json')}"
        fmachine = f"{SHORT_CMD}_{machine_file.split('.')[0]}.{jdata.get('pretty_format', 'json')}"

        try:
            logger.info(f"Pretty printing parameter data to {fparam}")
            dumpfn(jdata, fparam, indent=4)

            logger.info(f"Pretty printing machine data to {fmachine}")
            dumpfn(mdata, fmachine, indent=4)

        except Exception as e:
            logger.error(f"Error pretty printing data: {e}")

    if mdata.get("handlers", None) and mdata["handlers"].get("smtp", None):
        try:
            logger.info("Configuring SMTP handler for email notifications")

            que = queue.Queue(-1)
            queue_handler = logging.handlers.QueueHandler(que)
            smtp_handler = logging.handlers.SMTPHandler(**mdata["handlers"]["smtp"])
            listener = logging.handlers.QueueListener(que, smtp_handler)

            dlog.addHandler(queue_handler)

            logger.info("Starting SMTP listener")
            listener.start()

        except Exception as e:
            logger.error(f"Error configuring SMTP handler: {e}")

    try:
        logger.info("Converting machine data")
        mdata = convert_mdata(mdata)

    except Exception as e:
        logger.error(f"Error converting machine data: {e}")
        logger.error("Aborting due to error in machine data conversion")
        return

    max_tasks = 10000
    numb_task = 9
    record = "record.dpgen"
    iter_rec = [0, -1]

    if restart_task is not None:
        try:
            iter_rec = [int(restart_task.split('.')[0]), int(restart_task.split('.')[1])]
            logger.info(f"Restarting from iter {iter_rec[0]:03d} task {iter_rec[1]:02d}")
        except (IndexError, ValueError):
            logger.error(f"Invalid restart_task format: {restart_task}, expected ITER.TASK, e.g., 000000.06")
            return
    else:
        if os.path.isfile(record):
            try:
                logger.info(f"Reading record file: {record}")
                with open(record) as frec:
                    for line in frec:
                        iter_rec = [int(x) for x in line.split()]
                if len(iter_rec)==0:
                    raise RuntimeError("Blank lines found in record file")
                logger.info(f"Continuing from iter {iter_rec[0]:03d} task {iter_rec[1]:02d}")
            except Exception as e:
                logger.error(f"Error reading record file: {e}")
                logger.error("Aborting due to error in reading record file")
                return

    cont = True
    ii = iter_rec[0]

    while cont:
        iter_name = make_iter_name(ii)
        sepline(iter_name, "=")

        for jj in range(numb_task):
            if ii * max_tasks + jj <= iter_rec[0] * max_tasks + iter_rec[1]:
                logger.info(f"Skipping task {jj:02d} of iteration {ii:03d}, already completed")
                continue

            task_name = f"task {jj:02d}"
            sepline(f"{iter_name} {task_name}", "-")

            try:
                if jj == 0:
                    log_iter("make_train", ii, jj)
                    logger.info(f"Running task {jj:02d} (make_train) for iteration {ii:03d}")
                    make_train(ii, jdata, mdata)
                    logger.info(f"Task {jj:02d} (make_train) for iteration {ii:03d} completed")

                elif jj == 1:
                    log_iter("run_train", ii, jj)
                    logger.info(f"Running task {jj:02d} (run_train) for iteration {ii:03d}")
                    run_train(ii, jdata, mdata)
                    logger.info(f"Task {jj:02d} (run_train) for iteration {ii:03d} completed")

                elif jj == 2:
                    log_iter("post_train", ii, jj)
                    logger.info(f"Running task {jj:02d} (post_train) for iteration {ii:03d}")
                    post_train(ii, jdata, mdata)
                    logger.info(f"Task {jj:02d} (post_train) for iteration {ii:03d} completed")

                elif jj == 3:
                    log_iter("make_model_devi", ii, jj)
                    logger.info(f"Running task {jj:02d} (make_model_devi) for iteration {ii:03d}")
                    cont = make_model_devi(ii, jdata, mdata)
                    logger.info(f"Task {jj:02d} (make_model_devi) for iteration {ii:03d} completed")

                    if not cont:
                        logger.info(f"make_model_devi returned False, breaking loop")
                        break

                elif jj == 4:
                    log_iter("run_model_devi", ii, jj)
                    logger.info(f"Running task {jj:02d} (run_model_devi) for iteration {ii:03d}")
                    run_model_devi(ii, jdata, mdata)
                    logger.info(f"Task {jj:02d} (run_model_devi) for iteration {ii:03d} completed")

                elif jj == 5:
                    log_iter("post_model_devi", ii, jj)
                    logger.info(f"Running task {jj:02d} (post_model_devi) for iteration {ii:03d}")
                    post_model_devi(ii, jdata, mdata)
                    logger.info(f"Task {jj:02d} (post_model_devi) for iteration {ii:03d} completed")

                elif jj == 6:
                    log_iter("make_fp", ii, jj)
                    logger.info(f"Running task {jj:02d} (make_fp) for iteration {ii:03d}")
                    make_fp(ii, jdata, mdata)
                    logger.info(f"Task {jj:02d} (make_fp) for iteration {ii:03d} completed")

                elif jj == 7:
                    log_iter("run_fp", ii, jj)
                    logger.info(f"Running task {jj:02d} (run_fp) for iteration {ii:03d}")
                    run_fp(ii, jdata, mdata)
                    logger.info(f"Task {jj:02d} (run_fp) for iteration {ii:03d} completed")

                elif jj == 8:
                    log_iter("post_fp", ii, jj)
                    logger.info(f"Running task {jj:02d} (post_fp) for iteration {ii:03d}")
                    post_fp(ii, jdata)
                    logger.info(f"Task {jj:02d} (post_fp) for iteration {ii:03d} completed")

                else:
                    raise RuntimeError(f"unknown task {jj}")

                logger.info(f"Recording completed task {jj:02d} of iteration {ii:03d}")
                record_iter(record, ii, jj)

            except Exception as e:
                logger.error(f"Error during task {jj:02d} of iteration {ii:03d}: {e}", exc_info=True)
                logger.error(f"Aborting due to error in task {jj:02d} of iteration {ii:03d}")
                cont = False
                break

        ii += 1

    logger.info("All iterations completed, exiting")
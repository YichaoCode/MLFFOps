# config.py

import os
from dpgen import ROOT_PATH

TEMPLATE_NAME = "template"
TRAIN_NAME = "00.train"
TRAIN_TASK_FMT = "%03d"
TRAIN_TMPL_PATH = os.path.join(TEMPLATE_NAME, TRAIN_NAME)
DEFAULT_TRAIN_INPUT_FILE = "input.json"
DATA_SYSTEM_FMT = "%03d"
MODEL_DEVI_NAME = "01.model_devi"
MODEL_DEVI_TASK_FMT = DATA_SYSTEM_FMT + ".%06d"
MODEL_DEVI_CONF_FMT = DATA_SYSTEM_FMT + ".%04d"
FP_NAME = "02.fp"
FP_TASK_FMT = DATA_SYSTEM_FMT + ".%06d"
CVASP_FILE = os.path.join(ROOT_PATH, "generator/lib/cvasp.py")
CALYPSO_RUN_OPT_NAME = "gen_stru_analy"
CALYPSO_MODEL_DEVI_NAME = "model_devi_results"
CALYPSO_RUN_MODEL_DEVI_FILE = os.path.join(ROOT_PATH, "generator/lib/calypso_run_model_devi.py")
CHECK_OUTCAR_FILE = os.path.join(ROOT_PATH, "generator/lib/calypso_check_outcar.py")
RUN_OPT_FILE = os.path.join(ROOT_PATH, "generator/lib/calypso_run_opt.py")







template_name = "template"
train_name = "00.train"
train_task_fmt = "%03d"
train_tmpl_path = os.path.join(template_name, train_name)
default_train_input_file = "input.json"
data_system_fmt = "%03d"
model_devi_name = "01.model_devi"
model_devi_task_fmt = data_system_fmt + ".%06d"
model_devi_conf_fmt = data_system_fmt + ".%04d"
fp_name = "02.fp"
fp_task_fmt = data_system_fmt + ".%06d"
cvasp_file = os.path.join(ROOT_PATH, "generator/lib/cvasp.py")
# for calypso
calypso_run_opt_name = "gen_stru_analy"
calypso_model_devi_name = "model_devi_results"
calypso_run_model_devi_file = os.path.join(
    ROOT_PATH, "generator/lib/calypso_run_model_devi.py"
)
check_outcar_file = os.path.join(ROOT_PATH, "generator/lib/calypso_check_outcar.py")
run_opt_file = os.path.join(ROOT_PATH, "generator/lib/calypso_run_opt.py")

__all__ = [
    "TEMPLATE_NAME", "template_name",
    "TRAIN_NAME", "train_name",
    "TRAIN_TASK_FMT", "train_task_fmt",
    "TRAIN_TMPL_PATH", "train_tmpl_path",
    "DEFAULT_TRAIN_INPUT_FILE", "default_train_input_file",
    "DATA_SYSTEM_FMT", "data_system_fmt",
    "MODEL_DEVI_NAME", "model_devi_name",
    "MODEL_DEVI_TASK_FMT", "model_devi_task_fmt",
    "MODEL_DEVI_CONF_FMT", "model_devi_conf_fmt",
    "FP_NAME", "fp_name",
    "FP_TASK_FMT", "fp_task_fmt",
    "CVASP_FILE", "cvasp_file",
    "CALYPSO_RUN_OPT_NAME", "calypso_run_opt_name",
    "CALYPSO_MODEL_DEVI_NAME", "calypso_model_devi_name",
    "CALYPSO_RUN_MODEL_DEVI_FILE", "calypso_run_model_devi_file",
    "CHECK_OUTCAR_FILE", "check_outcar_file",
    "RUN_OPT_FILE", "run_opt_file",
]
# parse_utils.py

def parse_cur_job(cur_job):
    ensemble = _get_param_alias(cur_job, ["ens", "ensemble"])
    temps = [-1]
    press = [-1]
    if "npt" in ensemble:
        temps = _get_param_alias(cur_job, ["Ts", "temps"])
        press = _get_param_alias(cur_job, ["Ps", "press"])
    elif "nvt" == ensemble or "nve" == ensemble:
        temps = _get_param_alias(cur_job, ["Ts", "temps"])
    nsteps = _get_param_alias(cur_job, ["nsteps"])
    trj_freq = _get_param_alias(cur_job, ["t_freq", "trj_freq", "traj_freq"])
    if "pka_e" in cur_job:
        pka_e = _get_param_alias(cur_job, ["pka_e"])
    else:
        pka_e = None
    if "dt" in cur_job:
        dt = _get_param_alias(cur_job, ["dt"])
    else:
        dt = None
    return ensemble, nsteps, trj_freq, temps, press, pka_e, dt
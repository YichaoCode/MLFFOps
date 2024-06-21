# main.py
# created by Yichao
import os
import re
import shutil
import argparse
import logging
from workflow.run_iter import run_iter
# from workflow.run_iter_old import run_iter
from config.logger import setup_logging
from config.args import parse_args


def clean_up():
    """
    Clean up files and folders before running the program.
    """
    items_to_delete = [
        r"iter\.\d{6}",  # folders matching pattern iter.xxxxxx
        "work",  # folder
        "dpgen.log",  # file
        "dpdispatcher.log",  # file
        "record.dpgen"  # file
    ]

    for item in items_to_delete:
        if item == r"iter\.\d{6}":
            # Use regex to find all folders matching the pattern
            folders = [f for f in os.listdir('.') if re.match(item, f)]
            for folder in folders:
                try:
                    shutil.rmtree(folder)
                    logging.info(f"Deleted folder: {folder}")
                except Exception as e:
                    logging.warning(f"Failed to delete folder '{folder}': {e}")
        else:
            # For other items (files or folders), check directly
            if os.path.exists(item):
                if os.path.isdir(item):
                    try:
                        shutil.rmtree(item)
                        logging.info(f"Deleted folder: {item}")
                    except Exception as e:
                        logging.warning(f"Failed to delete folder '{item}': {e}")
                elif os.path.isfile(item):
                    try:
                        os.remove(item)
                        logging.info(f"Deleted file: {item}")
                    except Exception as e:
                        logging.warning(f"Failed to delete file '{item}': {e}")
            else:
                logging.info(f"Item not found: {item}")

def gen_run(args, clean=False):
    try:
        if args.param and args.machine:
            if clean:
                logging.debug("Cleaning up files and folders before running...")
                clean_up()  # Call the clean_up function if clean is True
            
            if args.debug:
                logging.getLogger("dflow").setLevel(logging.DEBUG)
            logging.info("Start running")
            run_iter(args.param, args.machine, restart_task=args.restart) 
            logging.info("Finished")
        else:
            logging.error("Bointh PARAM and MACHINE arguments are required.")
    except Exception as e:
        logging.error(f"Error during run: {e}", exc_info=True)

def main():
    args = parse_args()

    # 先配置日志
    custom_log_path = '//output/dpgen.log'
    logger = logging.getLogger('dpgen')
    logger.setLevel(logging.INFO)
    # logger.setLevel(logging.DEBUG)



    # 移除 dpgen logger 的现有处理程序
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(custom_log_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s-------------------------------------------------')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


    # 先配置日志
    custom_log_path = '//output/'
    dpgen_log_path = os.path.join(custom_log_path, 'dpgen.log')
    dpdispatcher_log_path = os.path.join(custom_log_path, 'dpdispatcher.log')


    # 配置 dpdispatcher logger
    dpdispatcher_logger = logging.getLogger('dpdispatcher')
    dpdispatcher_logger.setLevel(logging.INFO)
    # dpdispatcher_logger.setLevel(logging.DEBUG)
    for handler in dpdispatcher_logger.handlers[:]:
        dpdispatcher_logger.removeHandler(handler)
    dpdispatcher_file_handler = logging.FileHandler(dpdispatcher_log_path)
    dpdispatcher_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    dpdispatcher_file_handler.setFormatter(dpdispatcher_formatter)
    dpdispatcher_logger.addHandler(dpdispatcher_file_handler)



    # 然后调用其他初始化或配置函数
    setup_logging(args.debug)

    # 继续程序其余部分
    gen_run(args, clean=args.clean)

if __name__ == "__main__":
    main()

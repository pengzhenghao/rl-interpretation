import argparse
import os.path as osp
import subprocess
import sys
import time

JOB_NAME = "JOB_NAME"
NUM_NODES = "NUM_NODES"
NUM_CPUS_PER_NODE = "NUM_CPUS_PER_NODE"
NUM_GPUS_PER_NODE = "NUM_GPUS_PER_NODE"
PARTITION_NAME = "PARTITION_NAME"
NUM_WORKERS = "NUM_WORKERS"  # NUM_NODES - 1
COMMAND_PLACEHOLDER = "COMMAND_PLACEHOLDER"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="The job name and path to logging file (exp_name.log)."
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        required=True,
        help="Number of nodes to use."
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=64,
        help="Number of CPUs to use in each node. (Default: 64)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Number of GPUs to use in each node. (Default: 8)"
    )
    parser.add_argument(
        "--partition",
        "-p",
        type=str,
        default="VI_SP_Y_1080TI",
        help="Partition name. Default to VI_SP_Y_1080TI if in SH38 cluster."
    )
    parser.add_argument(
        "--command",
        type=str,
        required=True,
        help="The command you wish to execute. For example: --command 'python "
             "test.py' Note that the command must be a string."
    )
    args = parser.parse_args()

    job_name = "{}_{}".format(args.exp_name, time.strftime("%m%d-%H%M",
                                                           time.localtime()))

    # ===== Modified the template script =====
    with open(osp.join(osp.dirname(__file__), "template.txt"), "r") as f:
        text = f.read()
    text = text.replace(JOB_NAME, job_name)
    text = text.replace(NUM_NODES, str(args.num_nodes))
    text = text.replace(NUM_CPUS_PER_NODE, str(args.num_cpus))
    text = text.replace(NUM_GPUS_PER_NODE, str(args.num_gpus))
    text = text.replace(PARTITION_NAME, str(args.partition))
    text = text.replace(COMMAND_PLACEHOLDER, str(args.command))
    text = text.replace(NUM_WORKERS, str(args.num_nodes - 1))
    text = text.replace(
        "# THIS FILE IS A TEMPLATE AND IT SHOULD NOT BE DEPLOYED TO "
        "PRODUCTION!",
        "# THIS FILE IS MODIFIED AUTOMATICALLY FROM TEMPLATE AND SHOULD BE "
        "RUNNABLE!"
    )

    # ===== Save the script =====
    script_file = "{}.sh".format(job_name)
    with open(script_file, "w") as f:
        f.write(text)

    # ===== Submit the job =====
    print("Start to submit job!")
    subprocess.Popen(["sbatch", script_file])
    print("Job submitted! Script file is at: <{}>. Log file is at: <{}>".format(
        script_file, "{}.log".format(job_name)))
    sys.exit(0)

#!/usr/bin/env python

import os
from jobmonitor.api import register_job, upload_code_package
from jobmonitor.connections import mongo
import numpy as np


def upload_code():
    code_package, files_uploaded = upload_code_package(
        ".",
        excludes=[
            "core",
            "output.tmp",
            ".vscode",
            "benchmark",
            "node_modules",
            "scripts",
            ".git",
            "*.pyc",
            "__pycache__",
            "maintenance_*.py" "*.pdf",
            "*.js",
            "*.yaml",
            "._*",
            ".gitignore",
            ".AppleDouble",
        ],
    )
    print("Uploaded {} files.".format(len(files_uploaded)))

    return code_package


experiment = os.path.splitext(os.path.basename(__file__))[0]
project = "beyond-spectral-gap"
script = "train.py"
description = """
Trying to calibrate the random quadratics to Cifar-10 by changing the batch size.
We expect the rate to be r = eta^2 (1 + b + d) / b - 2 eta + 1, where b is the batch size
and want to fit 'd'.
""".strip()

code_package = upload_code()

base_config = {
    "seed": 1,
    "model_name": "VGG-11",
    "data_split_method": "random",
    "non_iid_alpha": None,
    "num_data_splits": None,
    "task": "Cifar",
    "num_epochs": 10,
    "batch_size": 32,
    "topology": "fully-connected",
    "momentum": 0.0,
    "weight_decay": 1e-4,
    "num_workers": 1,
    "eval_batch_size": 2000,
    "num_workers_to_eval": 2,
    "eval_period_start": 10,
    "eval_period_slowdown": 3,
}


for batch_size in [32, 64, 128, 256, 512, 1024]:
    for learning_rate in (0.5, 0.1, 0.01):
        config = {
            **base_config,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": 10 * batch_size // 32,
        }
        job_name = "batch{batch_size}-mom{momentum}-lr{learning_rate}".format(**config)

        if (
            mongo.job.count_documents(
                {
                    "job": job_name,
                    "experiment": experiment,
                    **{f"config.{key}": value for key, value in config.items()},
                }
            )
            > 0
        ):
            # We have this one already
            continue

        job_id = register_job(
            user="vogels",
            project=project,
            experiment=experiment,
            job=job_name,
            n_workers=1,
            priority=10,
            config_overrides=config,
            runtime_environment={
                "clone": {"code_package": code_package},
                "script": script,
            },
            annotations={"description": description},
        )
        print("{} - {}".format(job_id, job_name))
        # registered_ids.append(job_id)

#!/usr/bin/env python
import os

import numpy as np
from jobmonitor.api import register_job
from jobmonitor.api import upload_code_package
from jobmonitor.connections import mongo


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
    print(f"Uploaded {len(files_uploaded)} files.")

    return code_package


experiment = os.path.splitext(os.path.basename(__file__))[0]
project = "beyond-spectral-gap"
script = "train_iiddata.py"
description = """
We observed something strange. 32 workers with batch size 16 was different from 16 workers with batch size 32. Is that fixed now?
""".strip()

code_package = upload_code()

base_config = {
    "seed": 1,
    "model_name": "VGG-11",
    "data_split_method": "shared-data",
    "non_iid_alpha": None,
    "num_data_splits": None,
    "task": "Cifar",
    "num_epochs": 25,
    "batch_size": 16,
    "learning_rate": 0.03,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "num_workers": 32,
    "topology": "Fully connected",
    "num_workers_to_eval": 1,
    "eval_batch_size": 2000,
    "eval_alpha": 0.75,  # in [0.5, 1]. 0.5 means uniform, larger numbers focus alpha % of evaluation on the first half of each time window
    "eval_budget": 200,
    "ema_gamma": 0.95,
}

topo_names = [
    # "Time-varying exponential",
    # "Social network",
    # "Ring",
    # "Binary tree",
    # "Hypercube",
    # "Star",
    # "Torus (4x8)",
    # "Two cliques",
    # "Solo",
    "Fully connected",
]


registered_ids = []

for batch_size in [32, 16]:
    config = {**base_config, "batch_size": batch_size, "num_workers": 512 // batch_size}
    job_name = "n{num_workers}-{topology}-mom{momentum}-lr{learning_rate}".format(
        **config
    )

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
    print(f"jobrun_runai {job_id}")
    registered_ids.append(job_id)

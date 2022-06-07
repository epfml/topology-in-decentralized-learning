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
script = "train_compute_variance.py"
description = """
New approach to covariance between workers: we repeatedly store the same checkpoint and train for 100 steps. This gives us models across which we can compute covariance.
""".strip()

code_package = upload_code()

base_config = {
    "seed": 1,
    "model_name": "VGG-11",
    "non_iid_alpha": None,
    "num_data_splits": None,
    "task": "Cifar",
    "num_epochs": 10,
    "step_decay": [[1, 1]],
    "batch_size": 16,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "num_workers": 32,
    "num_workers_to_eval": 1,
    "eval_batch_size": 2000,
    "eval_alpha": 0.5,  # in [0.5, 1]. 0.5 means uniform, larger numbers focus alpha % of evaluation on the first half of each time window
    "eval_budget": 50,
    "ema_gamma": 0.0,
}

topos_and_lrs = [
    ("Binary tree", 0.0312),
    ("Fully connected", 0.0442),
    ("Hypercube", 0.0625),
    ("Ring", 0.0625),
    ("Social network", 0.0625),
    ("Solo", 0.0055),
    ("Star", 0.0156),
    ("Time-varying exponential", 0.0625),
    ("Torus (4x8)", 0.0625),
    ("Two cliques", 0.0442),
]


registered_ids = []

for (topology, learning_rate) in topos_and_lrs:
    config = {
        **base_config,
        "topology": topology,
        "learning_rate": learning_rate,
    }
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

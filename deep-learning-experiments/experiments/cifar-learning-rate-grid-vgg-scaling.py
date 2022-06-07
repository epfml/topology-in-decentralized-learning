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
script = "train.py"
description = """
Training Cifar with VGG (no batch norm) on many topologies and many learning rates.
Considering the early phase only, and using a constant learning rate.
Every node has access to all datapoints.
""".strip()

code_package = upload_code()

base_config = {
    "seed": 1,
    "model_name": "VGG-11",
    "data_split_method": "shared-data",
    "non_iid_alpha": None,
    "num_data_splits": None,
    "task": "Cifar",
    "num_epochs": 50,
    "batch_size": 32,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "num_workers": 32,
    "num_workers_to_eval": 2,
    "eval_batch_size": 2000,
    "eval_alpha": 0.75,  # in [0.5, 1]. 0.5 means uniform, larger numbers focus alpha % of evaluation on the first half of each time window
    "eval_budget": 100,
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

for num_workers in [2, 4, 8, 16]:
    for topology in topo_names:
        for learning_rate in np.logspace(start=-7, stop=0, base=2, num=10 + 1):
            learning_rate = round(learning_rate * 1000) / 1000
            config = {
                **base_config,
                "topology": topology,
                "num_epochs": int(50 * num_workers / 32),
                "num_workers": num_workers,
                "learning_rate": learning_rate,
            }
            job_name = (
                "n{num_workers}-{topology}-mom{momentum}-lr{learning_rate}".format(
                    **config
                )
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

    for topology in topo_names:
        for learning_rate in [0.001, 0.0001]:
            config = {
                **base_config,
                "topology": topology,
                "num_epochs": int(50 * num_workers / 32),
                "num_workers": num_workers,
                "learning_rate": learning_rate,
            }
            job_name = (
                "n{num_workers}-{topology}-mom{momentum}-lr{learning_rate:.4f}".format(
                    **config
                )
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

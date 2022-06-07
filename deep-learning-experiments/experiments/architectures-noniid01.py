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
            "maintenance_*.py"
            "*.pdf",
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
Running the same experiment on many topologies to investigate the impact on graph topology on quality. Tuning for learning rate.
Different from the previous experiment, we are now dividing the data non-iid.
""".strip()

code_package = upload_code()

base_config = {
    "seed": 1,
    "model_name": "VGG-11",
    "data_split_method": "dirichlet",
    "non_iid_alpha": 0.1,
    "num_data_splits": None,
    "task": "Cifar",
    "num_epochs": 100,
    "batch_size": 32,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "num_workers": 32,
    "eval_batch_size": 2000,
    "num_workers_to_eval": 2,
    "eval_period_start": 10,
    "eval_period_slowdown": 3,
}


topologies = [
    {"topology": "fully-connected", "topology_name": "fully-connected"},
    {"topology": "ring", "topology_name": "ring"},
    {"topology": "binary-tree", "topology_name": "binary-tree"},
    {"topology": "star", "topology_name": "star"},
    {"topology": "time-varying exponential", "topology_name": "sgp"},
    {"topology": "artificial-adversarial", "spectral_gap": 0.0128, "topology_name": "adv-0.0128"},
    {"topology": "artificial-nice", "spectral_gap": 0.0128, "topology_name": "nice-0.0128"},
    {"topology": "artificial-adversarial", "spectral_gap": 0.0312, "topology_name": "adv-0.0312"},
    {"topology": "artificial-nice", "spectral_gap": 0.0312, "topology_name": "nice-0.0312"},
    {"topology": "torus", "torus_side": 4, "topology_name": "torus-4-8"},
]

for spectral_gap in 10**np.linspace(-2, 0, 9):
    if spectral_gap == 1.0:
        continue
    spectral_gap = np.round(spectral_gap, decimals=4)
    topologies.append(
        {"topology": "artificial-adversarial", "spectral_gap": float(spectral_gap), "topology_name": f"adv-{spectral_gap}"},
    )

for spectral_gap in 10**np.linspace(-3.5, 0, 5):
    if spectral_gap == 1.0:
        continue
    spectral_gap = np.round(spectral_gap, decimals=4)
    topologies.append(
        {"topology": "artificial-nice", "spectral_gap": float(spectral_gap), "topology_name": f"nice-{spectral_gap}"},
    )


for spectral_gap in 10**np.linspace(-7, -3.5, 5):
    if spectral_gap == 1.0:
        continue
    spectral_gap = np.round(spectral_gap, decimals=5)
    topologies.append(
        {"topology": "artificial-nice", "spectral_gap": float(spectral_gap), "topology_name": f"nice-{spectral_gap}"},
    )


registered_ids = []

for topology in topologies:
    for learning_rate in (0.02,):
        for ema_gamma in (0,):
            config = {**base_config, **topology, "learning_rate": learning_rate, "ema_gamma": ema_gamma}
            job_name = "{topology_name}-mom{momentum}-lr{learning_rate}-gamma{ema_gamma}".format(**config)

            if mongo.job.count_documents({"job": job_name, "experiment": experiment, **{f"config.{key}": value for key, value in config.items()}}) > 0:
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
                runtime_environment={"clone": {"code_package": code_package}, "script": script},
                annotations={"description": description},
            )
            print("{} - {}".format(job_id, job_name))
            registered_ids.append(job_id)

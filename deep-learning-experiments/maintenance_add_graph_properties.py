import sys
from jobmonitor.connections import mongo
import torch
from bson import ObjectId
import topologies

sys.path.append("..")
import schemes

torch.set_default_dtype(torch.float64)

query = {
    "project": "beyond-spectral-gap", 
    "experiment": "architectures-noniid01",
    # "config.topology": "time-varying exponential"
}

projection = {
    "_id": True,
    "config": True,
}

def proposed_loss(params, scheme: schemes.AveragingScheme, up_to_power, regularization=0):
    n = scheme.n

    E = torch.ones([scheme.n, scheme.n]) / n
    total = 0.0

    for offset in range(scheme.period):
        Mpow = None
        for t in range(up_to_power + offset, 0 + offset, -1):
            w = scheme.w(t, params=params)
            if Mpow is None:
                Mpow = torch.eye(len(w))
            Mpow = Mpow @ w
            total += torch.sum((Mpow[:n, :n] - E)**2)
    
    reg = 0
    if regularization is not None and params is not None:
        reg = regularization * torch.abs(params).mean()

    return total / scheme.period + reg


for job in mongo.job.find(query, projection):
    # We have this one already
    config = job["config"]

    if config["topology"] == "time-varying exponential":
        scheme = schemes.TimeVaryingExponential(config["num_workers"])
        spectral_gap = -1
    else:
        gossip_matrix = topologies.configure_topology(**config).gossip_matrix().to(torch.float64)
        scheme = schemes.Matrix(gossip_matrix)
        spectral_gap = topologies.spectral_gap(gossip_matrix).item()
    
    if spectral_gap == -1:
        avg = 0
        for t in range(scheme.period):
            w = scheme.w(t)
            avg += w.T @ w
        avg /= scheme.period

        eig = torch.linalg.eig(avg)
        abs_eigenvalues = eig.eigenvalues.abs()
        sorted = torch.sort(abs_eigenvalues)
        abs_eigenvalues = sorted.values
        spectral_gap = 1 - abs_eigenvalues[-2].sqrt().item()
    
    random_walk_distance = proposed_loss(None, scheme, 10000).item()
    one_step_reduction = proposed_loss(None, scheme, 1).item()

    mongo.job.update_one({"_id": job["_id"]}, {"$set": {
        "config.topo_spectral_gap": spectral_gap,
        "config.topo_random_walk_distance": random_walk_distance,
        "config.topo_one_step_reduction": one_step_reduction,
    }})

    print(config["topology"], config["num_workers"], config.get("spectral_gap", ""), random_walk_distance, one_step_reduction, spectral_gap)

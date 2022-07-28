import numpy as np
from hyperopt import hp
from art.attacks.inference.model_inversion import MIFace
from counterfit.modules.algo.art.art import ArtInferenceAttack


class CFMIFace(ArtInferenceAttack):
    attack_cls = MIFace
    tags = ["image", "tabular"]
    category = "blackbox"

    parameters = {
        "default": {
            "max_iter":10000,
            "window_length":100,
            "threshold":0.99,
            "learning_rate":0.1,
            "batch_size":1,
            "verbose":True,
        },
        "optimize": {
            
            "max_iter":hp.uniform("mif_maxiter", 1000, 20000),
            "window_length": hp.uniform("mif_winlength", 10, 200),
            "threshold":hp.uniform("mif_threshold", 0.9, 0.9991),
            "learning_rate":hp.uniform("mif_lr",0.05,0.2),
        },
    }

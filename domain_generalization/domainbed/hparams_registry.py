# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np


def _hparams(algorithm, dataset, random_state):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ["Debug28", "RotatedMNIST", "ColoredMNIST"]

    hparams = {}

    hparams["data_augmentation"] = (True, True)
    hparams["val_augment"] = (False, False)  # augmentation for in-domain validation set
    hparams["resnet18"] = (False, False)
    hparams["resnet_dropout"] = (0.0, random_state.choice([0.0, 0.1, 0.5]))
    hparams["class_balanced"] = (False, False)
    hparams["optimizer"] = ("adam", "adam")

    hparams["freeze_bn"] = (True, True)
    hparams["pretrained"] = (True, True)  # only for ResNet

    
    if dataset not in SMALL_IMAGES:
        hparams["lr"] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
        if dataset == "DomainNet":
            hparams["batch_size"] = (32, int(2 ** random_state.uniform(3, 5)))
        else:
            hparams["batch_size"] = (32, int(2 ** random_state.uniform(3, 5.5)))
        if algorithm == "ARM":
            hparams["batch_size"] = (8, 8)
    else:
        hparams["lr"] = (1e-3, 10 ** random_state.uniform(-4.5, -2.5))
        hparams["batch_size"] = (64, int(2 ** random_state.uniform(3, 9)))

    if dataset in SMALL_IMAGES:
        hparams["weight_decay"] = (0.0, 0.0)
    else:
        hparams["weight_decay"] = (0.0, 10 ** random_state.uniform(-6, -2))

    if algorithm in ['FDANN']:
        hparams['disc_weight'] = (0.01, random_state.choice([0.01, 0.05, 0.1]))

        hparams['warm_up'] = (500, random_state.choice([300, 500, 1000]))
        hparams['maxinfo_weight'] = (0.0, random_state.choice([0.01, 0.1, 0.5]))

        hparams['ot_weight'] = (0.1, random_state.choice([0.01, 0.1, 0.5]))
        hparams['prototype_per_class'] = (16, random_state.choice([4, 8, 16, 32]))
        hparams['pro_weight'] = (0.01, random_state.choice([0.01, 0.1, 0.5]))
   

        hparams["mlp_width"] = (256, int(2 ** random_state.uniform(6, 10)))
        hparams["mlp_depth"] = (3, int(random_state.choice([3, 4, 5])))
        hparams["mlp_dropout"] = (0.5, random_state.choice([0.0, 0.1, 0.5]))

    if algorithm in ["DANN", "CDANN"]:
        if dataset not in SMALL_IMAGES:
            hparams["lr_g"] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
            hparams["lr_d"] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
        else:
            hparams["lr_g"] = (1e-3, 10 ** random_state.uniform(-4.5, -2.5))
            hparams["lr_d"] = (1e-3, 10 ** random_state.uniform(-4.5, -2.5))

        if dataset in SMALL_IMAGES:
            hparams["weight_decay_g"] = (0.0, 0.0)
        else:
            hparams["weight_decay_g"] = (0.0, 10 ** random_state.uniform(-6, -2))

        hparams["lambda"] = (0.01, 10 ** random_state.uniform(-2, 2))
        hparams["weight_decay_d"] = (0.0, 10 ** random_state.uniform(-6, -2))
        hparams["d_steps_per_g_step"] = (1, int(2 ** random_state.uniform(0, 3)))
        hparams["grad_penalty"] = (0.0, 10 ** random_state.uniform(-2, 1))
        hparams["beta1"] = (0.5, random_state.choice([0.0, 0.5]))
        hparams["mlp_width"] = (256, int(2 ** random_state.uniform(6, 10)))
        hparams["mlp_depth"] = (3, int(random_state.choice([3, 4, 5])))
        hparams["mlp_dropout"] = (0.0, random_state.choice([0.0, 0.1, 0.5]))
   
    elif algorithm in ["IRM_DANN", "IRM_CDANN"]:
        # DANN and CDANN
        if dataset not in SMALL_IMAGES:
            hparams["lr_g"] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
            hparams["lr_d"] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
        else:
            hparams["lr_g"] = (1e-3, 10 ** random_state.uniform(-4.5, -2.5))
            hparams["lr_d"] = (1e-3, 10 ** random_state.uniform(-4.5, -2.5))

        if dataset in SMALL_IMAGES:
            hparams["weight_decay_g"] = (0.0, 0.0)
        else:
            hparams["weight_decay_g"] = (0.0, 10 ** random_state.uniform(-6, -2))

        hparams["lambda"] = (1.0, 10 ** random_state.uniform(-2, 2))
        hparams["weight_decay_d"] = (0.0, 10 ** random_state.uniform(-6, -2))
        hparams["d_steps_per_g_step"] = (1, int(2 ** random_state.uniform(0, 3)))
        hparams["grad_penalty"] = (0.0, 10 ** random_state.uniform(-2, 1))
        hparams["beta1"] = (0.5, random_state.choice([0.0, 0.5]))
        hparams["mlp_width"] = (256, int(2 ** random_state.uniform(6, 10)))
        hparams["mlp_depth"] = (3, int(random_state.choice([3, 4, 5])))
        hparams["mlp_dropout"] = (0.0, random_state.choice([0.0, 0.1, 0.5]))

    elif algorithm == 'IIB':
        hparams['lambda_beta'] =  (1e-4, lambda r: 1e-3 ** r.uniform(-2, 2))
        hparams['lambda_inv_risks'] = (10, lambda r: int(10 ** r.uniform(0, 2)))
        hparams['enable_bn'] = (True, lambda r: bool(r.choice([True, True])))

    elif algorithm == "RSC":
        hparams["rsc_f_drop_factor"] = (1 / 3, random_state.uniform(0, 0.5))
        hparams["rsc_b_drop_factor"] = (1 / 3, random_state.uniform(0, 0.5))
    elif algorithm == "SagNet":
        hparams["sag_w_adv"] = (0.1, 10 ** random_state.uniform(-2, 1))
    elif algorithm == "IRM":
        hparams["irm_lambda"] = (1e2, 10 ** random_state.uniform(-1, 5))
        hparams["irm_penalty_anneal_iters"] = (
            500,
            int(10 ** random_state.uniform(0, 4)),
        )

        

    elif algorithm in ["Mixup", "OrgMixup"]:
        hparams["mixup_alpha"] = (0.2, 10 ** random_state.uniform(-1, -1))
    elif algorithm == "GroupDRO":
        hparams["groupdro_eta"] = (1e-2, 10 ** random_state.uniform(-3, -1))
    elif algorithm in ("MMD", "CORAL"):
        hparams["mmd_gamma"] = (1.0, 10 ** random_state.uniform(-1, 1))
    elif algorithm in ("MLDG", "SOMLDG"):
        hparams["mldg_beta"] = (1.0, 10 ** random_state.uniform(-1, 1))
    elif algorithm == "MTL":
        hparams["mtl_ema"] = (0.99, random_state.choice([0.5, 0.9, 0.99, 1.0]))
    elif algorithm == "VREx":
        hparams["vrex_lambda"] = (1e1, 10 ** random_state.uniform(-1, 5))
        hparams["vrex_penalty_anneal_iters"] = (
            500,
            int(10 ** random_state.uniform(0, 4)),
        )
    elif algorithm == "SAM":
        hparams["rho"] = (0.05, random_state.choice([0.01, 0.02, 0.05, 0.1]))
    elif algorithm == "CutMix":
        hparams["beta"] = (1.0, 1.0)
        # cutmix_prob is set to 1.0 for ImageNet and 0.5 for CIFAR100 in the original paper.
        hparams["cutmix_prob"] = (1.0, 1.0)
    elif algorithm == "SD":
        hparams['sd_reg'] = (0.1, lambda r: 10**r.uniform(-5, -1))

    return hparams


def default_hparams(algorithm, dataset):
    dummy_random_state = np.random.RandomState(0)
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, dummy_random_state).items()}


def random_hparams(algorithm, dataset, seed):
    random_state = np.random.RandomState(seed)
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, random_state).items()}


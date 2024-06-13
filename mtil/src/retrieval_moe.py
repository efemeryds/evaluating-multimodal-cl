""" Running retrieval evaluation on checkpoints of choice"""
import os
import clip_moe
import torch
from . import utils
from .args import parse_arguments
from .models import retrieval_evaluation_moe, Autoencoder, Alexnet_FE
import torchvision.models as models
import torch.nn as nn


def main(input_arguments):
    print("input_arguments type", type(input_arguments))
    # sets a random seed for all relevant libraries
    utils.seed_all(input_arguments.seed)

    if input_arguments.retrieval:
        # loads clip model
        print("Loading clip model...")
        model, _, val_preprocess = clip_moe.load(input_arguments.model, jit=False, args=input_arguments)
        print("Finished loading clip model..")
        # loads current checkpoint for the evaluation
        if (input_arguments.load_autochooser and input_arguments.autorouter is True) and input_arguments.load:
            utils.torch_load(model, input_arguments.load)
            # a feature extractor model from an Alexnet architecture, that is used to train the autoencoder model
            # and get the most related model whilst training a new task in a sequence
            pretrained_alexnet = models.alexnet(pretrained=True)
            if torch.cuda.is_available():
                feature_extractor = Alexnet_FE(pretrained_alexnet).cuda()
            else:
                feature_extractor = Alexnet_FE(pretrained_alexnet).cpu()
            # storing a list of modules
            autoencoder_list = nn.ModuleList()
            # preparing Autoencoders so that the weights from the checkpoint file are loaded into them
            for i in range(input_arguments.task_num + 1):
                model_autoencoder = Autoencoder(256 * 13 * 13)
                autoencoder_list.append(model_autoencoder)
            utils.torch_load(autoencoder_list, input_arguments.load_autochooser)
            if torch.cuda.is_available():
                autoencoder_list = autoencoder_list.cuda()
            else:
                autoencoder_list = autoencoder_list.cpu()

            retrieval_evaluation_moe(model, feature_extractor, autoencoder_list, input_arguments, val_preprocess)
        else:
            return


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

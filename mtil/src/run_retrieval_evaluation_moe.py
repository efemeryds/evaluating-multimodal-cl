import os
import clip_moe
import torch
from . import utils
from .args import parse_arguments
from .models import retrieval_evaluation_moe, Autoencoder, Alexnet_FE
import torchvision.models as models
import torch.nn as nn


def main(args):
    utils.seed_all(args.seed)

    if args.eval_only:
        model, _, val_preprocess = clip_moe.load(args.model, jit=False, args=args)
        if args.load:
            utils.torch_load(model, args.load)
        if args.load_autochooser and args.autorouter is True:
            pretrained_alexnet = models.alexnet(pretrained=True)
            if torch.cuda.is_available():
                feature_extractor = Alexnet_FE(pretrained_alexnet).cuda()
            else:
                feature_extractor = Alexnet_FE(pretrained_alexnet).cpu()
            autoencoder_list = nn.ModuleList()
            for i in range(args.task_num + 1):
                model_autoencoder = Autoencoder(256 * 13 * 13)
                autoencoder_list.append(model_autoencoder)
            utils.torch_load(autoencoder_list, args.load_autochooser)
            if torch.cuda.is_available():
                autoencoder_list = autoencoder_list.cuda()
            else:
                autoencoder_list = autoencoder_list.cpu()
        elif args.save:
            checkpoint_pth = os.path.join(
                args.save, f"clip_zeroshot_{args.train_dataset}.pth"
            )
            utils.torch_save(checkpoint_pth, model)
        retrieval_evaluation_moe(model, feature_extractor, autoencoder_list, args, val_preprocess)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

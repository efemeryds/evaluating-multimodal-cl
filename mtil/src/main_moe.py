import os
import clip_moe
import torch
from . import utils
from .args import parse_arguments
from .models import evaluate_moe, finetune_moe, Autoencoder, Alexnet_FE, few_shot_AutoEncoder, AutoEncoder
import torchvision.models as models
import torch.nn as nn


def main(args):
    utils.seed_all(args.seed)

    assert args.train_mode in ["whole", "text", "image", "adapter"]
    if args.eval_only:
        model, _, val_preprocess = clip_moe.load(args.model, jit=False, args=args)
        if args.load:  #
            utils.torch_load(model, args.load)
        if args.load_autochooser and args.autorouter == True:
            pretrained_alexnet = models.alexnet(pretrained=True)
            if torch.cuda.is_available():
                feature_extractor = Alexnet_FE(pretrained_alexnet).cuda()
            else:
                feature_extractor = Alexnet_FE(pretrained_alexnet).cpu()
            autoencoder_list = nn.ModuleList()
            for i in range(args.task_num + 1):  # more for zero-shot chosen  / few or full shot share the code
                model_autoencoder = Autoencoder(256 * 13 * 13)
                autoencoder_list.append(model_autoencoder)
            utils.torch_load(autoencoder_list, args.load_autochooser)
            if torch.cuda.is_available():
                autoencoder_list = autoencoder_list.cuda()
            else:
                autoencoder_list = autoencoder_list.cpu()
        elif args.save:  # None
            checkpoint_pth = os.path.join(
                args.save, f"clip_zeroshot_{args.train_dataset}.pth"
            )
            utils.torch_save(checkpoint_pth, model)
        evaluate_moe(model, feature_extractor, autoencoder_list, args, val_preprocess)

    else:
        if args.train_chooser:
            if args.few_shot > 0:
                print('----------------------train few-shot chooser----------------------')
                chooser_of_few_shot = few_shot_AutoEncoder(args)  # few shot chooser
            else:
                print('----------------------train full-shot chooser----------------------')
                chooser = AutoEncoder(args)
        else:
            print('----------------------finetune model----------------------')
            print(args)
            model = finetune_moe(args)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

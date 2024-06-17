import clip_moe
import torch
import os
from torch.utils.data import default_collate
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from .. import datasets
from ..datasets.common import get_dataloader, maybe_dictionarize
import torch.nn.functional as F
from .AutoEncoder import encoder_criterion
from .zeroshot_retrieval_laion_ai import evaluate
from clip_moe.tokenizer import SimpleTokenizer
from ..datasets import multilingual_mscoco


def retrieval_evaluation_moe(image_classifier: clip_moe.model.CLIP, feature_extractor,
                             autoencoder_list: torch.nn.modules.container.ModuleList, args,
                             val_preprocess: torchvision.transforms.transforms.Compose) -> None:
    if args.eval_datasets is None:
        return
    for i, dataset_name in enumerate(args.eval_datasets):
        root = "/home/alicja/repositories/multi-modal-continual-learning/mtil/tmp_data"
        language = "en"
        annotation_file = os.path.join(root, multilingual_mscoco.OUTPUT_FILENAME_TEMPLATE.format(language))
        if not os.path.exists(annotation_file):
            multilingual_mscoco.create_annotation_file(root, language)

        dataset = multilingual_mscoco.Multilingual_MSCOCO(root=root, ann_file=annotation_file, transform=val_preprocess)

        print("Evaluating on: ", dataset_name)

        dataset_class = getattr(datasets, dataset_name)
        # dataset = dataset_class(
        #     val_preprocess,
        #     location=args.data_location,
        #     batch_size=args.batch_size,
        #     batch_size_eval=args.batch_size_eval,
        # )
        #dataset = dataset_class(val_preprocess,
         #                       ann_file="/home/alicja/repositories/multi-modal-continual-learning/mtil/tmp_data/multilingual_mscoco_captions-en.json")

        retrieval_evaluation_of_single_dataset(image_classifier, feature_extractor, autoencoder_list, dataset, args)


def retrieval_evaluation_of_single_dataset(image_classifier, feature_extractor, autoencoder_list, dataset, args):
    model = image_classifier
    input_key = "images"
    image_enc = None

    autoencoder_list.eval()
    model.eval()
    # zeroshot_weights = zeroshot_classifier(
    #     dataset.classnames, dataset.templates, model, args
    # )

    # dataloader = get_dataloader(
    #     dataset, is_train=False, args=args, image_encoder=image_enc
    # )

    # transform = transforms.ToTensor()

    def image_captions_collate_fn(batch):
        transposed = list(zip(*batch))
        print(type(transposed))
        # imgs = default_collate(transposed)
        # texts = transposed[1]
        imgs = transposed[0]
        texts = transposed[1]
        # Apply the default collate function to images (this converts them to tensors)
        imgs = default_collate(imgs)

        # Return images and texts as separate entities
        return imgs, texts
        # return imgs, texts

    # print(dataset)

    collate_fn = image_captions_collate_fn(dataset)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8,
        shuffle=False, num_workers=4
    )

    print("Starting zeroshot eval..")
    # top1, top5 = zeroshot_eval(model, feature_extractor, autoencoder_list, dataloader, zeroshot_weights, args)

    device = 'cuda'
    # TODO: ensure that as in standard evaluation I need to identify a task using autoencoders first
    zeroshot_retrieval_eval(model, feature_extractor, autoencoder_list, dataloader, args)

    # TODO: use clip benchmark main code and add to readme
    top1 = ''
    top5 = ''

    print("Finishing zeroshot eval..")

    print(f"Top-1 accuracy: {top1:.2f}")
    return


@torch.no_grad()
def zeroshot_retrieval_eval(model, feature_extractor, autoencoder_list, loader, args):
    top1, top5, n = 0.0, 0.0, 0.0
    task_id = 0
    print("loader type", type(loader))
    print(loader)
    # for i, data in enumerate(tqdm(loader)):
    #     print("data type", type(data))
    #     print("i type", type(i))
    #     data = maybe_dictionarize(data)
    #     if torch.cuda.is_available():
    #         images = data["images"].cuda()
    #         target = data["labels"].cuda()
    #     else:
    #         images = data["images"].cpu()
    #         target = data["labels"].cpu()
    #
    #     # predict batch image domain:
    #     input_to_ae = feature_extractor(images)
    #     input_to_ae = input_to_ae.view(input_to_ae.size(0), -1)
    #     input_to_ae = input_to_ae
    #     input_to_ae = F.sigmoid(input_to_ae)  # GT
    #
    #     # print("input_to_ae", input_to_ae)
    #     model_autoencoder = autoencoder_list[0]
    #     outputs = model_autoencoder(input_to_ae)
    #     # print("outputs", outputs)
    #     best_loss = encoder_criterion(outputs, input_to_ae)
    #     # print("best l", best_loss)
    #     best_router = 0
    #     for i in range(1, 12):
    #         outputs = autoencoder_list[i](input_to_ae)
    #         new_loss = encoder_criterion(outputs, input_to_ae)
    #         # print("new loss", new_loss)
    #         if new_loss < best_loss:
    #             best_loss = new_loss
    #             best_router = i
    #         if best_loss > args.threshold:
    #             best_router = 0
    #     task_id = best_router - 1
    #     # predict
    #     image_features = model.encode_image(images, task_id)
    #     image_features /= image_features.norm(dim=-1, keepdim=True)
    # logits = 100.0 * image_features @ zeroshot_weights[task_id + 1]
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # TODO: update here?
    metrics = evaluate(model, loader, SimpleTokenizer, device, task_id)
    metrics.to_csv("metrics.csv")

    # acc1, acc5 = accuracy(logits, target, topk=(1, 5))
    # top1 += acc1
    # top5 += acc5

    # n += images.size(0)

    # top1 = (top1 / n) * 100
    # top5 = (top5 / n) * 100
    # return top1, top5
    return


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    # print('pred',pred)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


@torch.no_grad()
def zeroshot_classifier(classnames, templates, model, args):
    if not isinstance(templates, list):
        templates = [templates]
    zeroshot_weights = []
    # for task_id in range(args.task_num):
    for task_id in range(-1, args.task_num):
        zeroshot_weights_i = []
        for classname in classnames:
            texts = [template(classname) for template in templates]  # format with class
            if torch.cuda.is_available():
                texts = clip_moe.tokenize(texts).cuda()
            else:
                texts = clip_moe.tokenize(texts).cpu()
            # tokenize
            if args.non_text == True:
                class_embeddings = model.encode_text(texts, -1)  # embed with text encoder
            else:
                class_embeddings = model.encode_text(texts, task_id)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights_i.append(class_embedding)
        if torch.cuda.is_available():
            zeroshot_weights_i = torch.stack(zeroshot_weights_i, dim=1).cuda()
        else:
            zeroshot_weights_i = torch.stack(zeroshot_weights_i, dim=1).cpu()

        zeroshot_weights.append(zeroshot_weights_i)
    return zeroshot_weights


@torch.no_grad()
def zeroshot_eval(model, feature_extractor, autoencoder_list, loader, zeroshot_weights, args):
    top1, top5, n = 0.0, 0.0, 0.0
    print("loader type", type(loader))
    for i, data in enumerate(tqdm(loader)):
        print("data type", type(data))
        print("i type", type(i))
        data = maybe_dictionarize(data)
        if torch.cuda.is_available():
            images = data["images"].cuda()
            target = data["labels"].cuda()
        else:
            images = data["images"].cpu()
            target = data["labels"].cpu()

        # predict batch image domain:
        input_to_ae = feature_extractor(images)
        input_to_ae = input_to_ae.view(input_to_ae.size(0), -1)
        input_to_ae = input_to_ae
        input_to_ae = F.sigmoid(input_to_ae)  # GT

        # print("input_to_ae", input_to_ae)
        model_autoencoder = autoencoder_list[0]
        outputs = model_autoencoder(input_to_ae)
        # print("outputs", outputs)
        best_loss = encoder_criterion(outputs, input_to_ae)
        # print("best l", best_loss)
        best_router = 0
        for i in range(1, 12):
            outputs = autoencoder_list[i](input_to_ae)
            new_loss = encoder_criterion(outputs, input_to_ae)
            # print("new loss", new_loss)
            if new_loss < best_loss:
                best_loss = new_loss
                best_router = i
            if best_loss > args.threshold:
                best_router = 0
        task_id = best_router - 1
        # predict
        image_features = model.encode_image(images, task_id)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ zeroshot_weights[task_id + 1]
        # [zeroshot_weights]:-1 to 11
        # measure accuracy
        # print("target", target)
        # print("logits", logits)
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        # print("images.size(0)", images.size(0))
        n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    return top1, top5

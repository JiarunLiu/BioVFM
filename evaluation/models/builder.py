
import os
import torch
from functools import partial

from util import interpolate_pos_embed
from timm.models.layers import trunc_normal_

from .biomedclip import biomedclip_vit_base_patch16
from .dinov2 import dino_vit_tiny, dino_vit_small, dino_vit_base, dino_vit_large, dino_vit_huge, dino_vit_giant2, DEFAULT_DINOV2_KWARGS
from .models_vit import vit_tiny_patch16, vit_small_patch16, vit_base_patch16, vit_large_patch16, vit_huge_patch14



AVAILABLE_MODELS = {
    # ViT models
    "vit_tiny_patch16": vit_tiny_patch16,
    "vit_small_patch16": vit_small_patch16,
    "vit_base_patch16": vit_base_patch16,
    "vit_large_patch16": vit_large_patch16,
    "vit_huge_patch14": vit_huge_patch14,
    # biomedclip, it is ViT-B actrually
    "biomedclip_vit_base_patch16": biomedclip_vit_base_patch16,
    # dino models
    "dino_vit_tiny": dino_vit_tiny,
    "dino_vit_small": dino_vit_small,
    "dino_vit_base": dino_vit_base,
    "dino_vit_base_reg4": partial(dino_vit_base, num_register_tokens=4),
    "dino_vit_large": dino_vit_large,
    "dino_vit_huge": dino_vit_huge,
}



def build_vit_model(model_flag, n_classes, global_pool, pretrain_model, enable_finetune):

    use_imagenet_pretrain = False
    if pretrain_model.lower() == "imagenet":
        assert model_flag in ['vit_base_patch16', 'vit_large_patch16']
        use_imagenet_pretrain = True

    # model = models_vit.__dict__[model_flag](
    model = AVAILABLE_MODELS[model_flag](
        num_classes=n_classes,
        global_pool=global_pool,
        pretrained=use_imagenet_pretrain,
    )

    # load pretrain checkpoint
    if os.path.isfile(pretrain_model):
        checkpoint = torch.load(pretrain_model, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % pretrain_model)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)
    else:
        if not use_imagenet_pretrain:
            print("Pretrain model checkpoint file does not exist!")
            print("Initialize model with random weights.")

    # employ linear probing by default
    if enable_finetune:
        print("Enable finetune mode: all parameters are trainable.")
    else:
        print("Enable linear probing mode: only head parameters are trainable.")
        # for linear prob only
        # hack: revise model's head with BN
        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
        # freeze all but the head
        for _, p in model.named_parameters():
            p.requires_grad = False
        for _, p in model.head.named_parameters():
            p.requires_grad = True

    return model


def build_biomedclip_model(model_flag, n_classes, global_pool, pretrain_model, enable_finetune):
    assert model_flag == "biomedclip_vit_base_patch16"

    use_imagenet_pretrain = False
    if pretrain_model.lower() == "imagenet":
        assert model_flag in ['vit_base_patch16', 'vit_large_patch16']
        use_imagenet_pretrain = True

    model = biomedclip_vit_base_patch16(
        num_classes=n_classes,
        global_pool=global_pool,
        pretrained=use_imagenet_pretrain,
    )

    # load pretrain checkpoint
    if os.path.isfile(pretrain_model):
        checkpoint = torch.load(pretrain_model, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % pretrain_model)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)
    else:
        if not use_imagenet_pretrain:
            print("Pretrain model checkpoint file does not exist!")
            print("Initialize model with random weights.")

    # employ linear probing by default
    if enable_finetune:
        print("Enable finetune mode: all parameters are trainable.")
    else:
        print("Enable linear probing mode: only head parameters are trainable.")
        # for linear prob only
        # hack: revise model's head with BN
        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
        # freeze all but the head
        for _, p in model.named_parameters():
            p.requires_grad = False
        for _, p in model.head.named_parameters():
            p.requires_grad = True

    return model


def load_dino_pretrained_weights(model, pretrained_weights, checkpoint_key):
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        # logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)


def build_dino_model(model_flag, n_classes, global_pool, pretrain_model, enable_finetune, **kwargs):
    # check the needs of load imagenet model
    use_imagenet_pretrain = False
    if pretrain_model.lower() == "imagenet":
        raise NotImplementedError("DINO model does not support imagenet pretrain model. Please use ViT instead.")
    
    # add default kwargs
    for k, v in DEFAULT_DINOV2_KWARGS.items():
        if k not in kwargs:
            kwargs[k] = v

    # patch_size is 16 by default, except for huge model
    if model_flag == 'dino_vit_huge':
        kwargs['patch_size'] = 14
        kwargs['drop_path_rate'] = 0.4
        kwargs['ffn_layer'] = "swiglufused"
        kwargs['block_chunks'] = 4

    model = AVAILABLE_MODELS[model_flag](
        num_classes=n_classes,
        global_pool=global_pool,
        **kwargs,
    )

    # load pretrain checkpoint
    if os.path.isfile(pretrain_model):
        checkpoint = torch.load(pretrain_model, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % pretrain_model)
        checkpoint_model = checkpoint['teacher']
        # remove `module.` prefix
        checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint_model.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)
    else:
        print("Pretrain model checkpoint file does not exist!")
        print("Initialize model with random weights.")

    # employ linear probing by default
    if enable_finetune:
        print("Enable finetune mode: all parameters are trainable.")
    else:
        print("Enable linear probing mode: only head parameters are trainable.")
        # hack: revise model's head with BN
        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
        # freeze all but the head
        for _, p in model.named_parameters():
            p.requires_grad = False
        for _, p in model.head.named_parameters():
            p.requires_grad = True

    return model



def build_model(model_flag, n_class, pretrain_model, enable_finetune, **kwargs):

    if "dino_vit" in model_flag:
        model = build_dino_model(
            model_flag=model_flag,
            n_classes=n_class,
            global_pool=kwargs.get('global_pool', False),
            pretrain_model=pretrain_model,
            enable_finetune=enable_finetune)
    elif "vit" in model_flag:
        model = build_vit_model(
            model_flag=model_flag,
            n_classes=n_class,
            global_pool=kwargs.get('global_pool', False),
            pretrain_model=pretrain_model,
            enable_finetune=enable_finetune,
        )
    elif "biomedclip" in model_flag:
        model = build_biomedclip_model(
            model_flag=model_flag,
            n_classes=n_class,
            global_pool=kwargs.get('global_pool', False),
            pretrain_model=pretrain_model,
            enable_finetune=enable_finetune,
        )
    else:
        raise NotImplementedError


    return model
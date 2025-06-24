import timm


def biomedclip_vit_base_patch16(pretrained=False, **kwargs):
    assert timm.__version__ >= "0.9.8", "Please upgrade timm to 0.4.5 or later for BiomedCLIP"
    from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8

    model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', 
        cache_dir="/home/fgldlb/disk2/model_zoo"
    )
    vision_model = model.visual.trunk
    # reset the head
    global_pool = kwargs.get('global_pool', True)
    vision_model.reset_classifier(
        num_classes=kwargs.get('num_classes', 10),
        global_pool="avg" if global_pool else "token",
    )

    return vision_model
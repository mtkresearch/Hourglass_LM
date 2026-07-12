from hydra.utils import instantiate


def create_dataloader(data_cfg, is_training=True):
    transform = instantiate(data_cfg.transform, is_training=is_training)
    print(f'Data augmentation is as follows \n{transform}\n')

    if is_training:
        dataset = instantiate(data_cfg.trainset, transform=transform)
        print(f'{len(dataset)} images and {data_cfg.baseinfo.num_classes} classes were found from {data_cfg.trainset.root}')
    else:
        dataset = instantiate(data_cfg.valset, transform=transform)
        print(f'{len(dataset)} images and {len(dataset.classes)} classes were found from {data_cfg.valset.root}')

    sampler = instantiate(data_cfg.sampler, dataset=dataset, shuffle=is_training)
    dataloader = instantiate(data_cfg.loader, dataset=dataset, sampler=sampler, drop_last=is_training)
    return dataloader

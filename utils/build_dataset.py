import os
import torch
from data.transform import Resize, ToTensor
from data import dataset
from data.batch_sampler import TwoStreamBatchSampler
from test.eval import ImageFolderDataset
from torchvision import transforms
import json


def split_csv_or_list(val):
    """
    Normalize config values that may be a YAML list or a comma-separated string.
    Returns a list of non-empty stripped strings.
    """
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        return [p.strip() for p in val.split(",") if p.strip()]
    return [str(val).strip()]


# Helper function to get the number of labeled images
def _labeled_num(train_num, cfg):
    if cfg.get('data.label_mode') == 'percentage':
        n = round(train_num * cfg.get('data.labeled_perc') / 100)
    elif cfg.get('data.label_mode') == 'number':
        n = cfg.get('data.labeled_num')
    else:
        raise ValueError("data.label_mode must be 'percentage' or 'number'")
    if n % 2 != 0:
        n -= 1
    return n

def _get_list_name(cfg):
    json_filename = cfg.get('data.json_filename', None)
    if json_filename is not None:
        with open(json_filename, 'r') as f:
            list_name = json.load(f)
    else:
        list_name = None
    return list_name

def build_dataset(cfg):
    val_perc = int(cfg.get('data.val_split_perc', 0))
    resize_h = cfg.get('data.eval.resize_height', 320)
    resize_w = cfg.get('data.eval.resize_width', 320)
    val_test_transform = transforms.Compose([
        Resize((resize_w, resize_h)),
        ToTensor()
    ])

    train_dataloader, val_dataloader = None, None
    if val_perc == 0:
        list_name = _get_list_name(cfg)

        train_data = getattr(dataset, cfg.get('data.dataset'))(
            root=cfg.get('data.root'), data2_dir=cfg.get('data.data2_dir'),
            mode='train', require_depth=cfg.get('data.require_depth'), 
            depth_dirname=cfg.get('data.depth_dirname', None), list_name=list_name)
        train_num = len(train_data)
        print(f"Total training images: {train_num}")
        labeled_num = _labeled_num(train_num, cfg)
        print(f"Labelled images: {labeled_num}")
        print(f"Unlabelled images: {train_num - labeled_num}")
        batch_sampler = TwoStreamBatchSampler(
            train_num, labeled_num,
            int(cfg.get('data.labeled_bs')), int(cfg.get('data.batch_size')) - int(cfg.get('data.labeled_bs')))
        train_dataloader = torch.utils.data.DataLoader(
            train_data, batch_sampler=batch_sampler,
            shuffle=cfg.get('data.shuffle'), num_workers=cfg.get('data.num_workers'))

    elif val_perc < 100:
        print(f"Validation split: {val_perc}%")
        data_root = os.path.join(cfg.get('data.root'), cfg.get('data.data2_dir'), cfg.get('data.image_dirname'))
        list_name = _get_list_name(cfg)

        if list_name is None:
            list_name = sorted(
                f
                for f in os.listdir(data_root)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            )
        total_num = len(list_name)
        val_num = round(total_num * val_perc / 100)
        train_num = total_num - val_num
        print(f"Total training images: {train_num}, validation images: {val_num}")
        train_files = list_name[:train_num]
        val_files = list_name[train_num:]
        train_data = getattr(dataset, cfg.get('data.dataset'))(
            root=cfg.get('data.root'), data2_dir=cfg.get('data.data2_dir'),
            mode='train', require_depth=cfg.get('data.require_depth'),
            image_dirname=cfg.get('data.image_dirname'),
            mask_dirname=cfg.get('data.mask_dirname'),
            depth_dirname=cfg.get('data.depth_dirname', None),
             list_name=train_files)
        labeled_num = _labeled_num(len(train_files), cfg)


        print(f"Total training images: {train_num}, labelled: {labeled_num} ({labeled_num / train_num * 100:.2f}%)")
        batch_sampler = TwoStreamBatchSampler(
            train_num, labeled_num,
            int(cfg.get('data.labeled_bs')), int(cfg.get('data.batch_size')) - int(cfg.get('data.labeled_bs')))
        train_dataloader = torch.utils.data.DataLoader(
            train_data, batch_sampler=batch_sampler,
            shuffle=cfg.get('data.shuffle'), num_workers=cfg.get('data.num_workers'))
        train_dataset_root = os.path.join(cfg.get('data.root'), cfg.get('data.data2_dir'))
        val_data = ImageFolderDataset(
            dataset_root=train_dataset_root,
            image_dirname=cfg.get('data.test.image_dirname'),
            mask_dirname=cfg.get('data.test.mask_dirname'),
            depth_dirname=cfg.get('data.test.depth_dirname', None),
            transform=val_test_transform, list_name=val_files)
        val_dataloader = torch.utils.data.DataLoader(
            val_data, batch_size=cfg.get('data.test.batch_size'), shuffle=False, num_workers=0)
    else:
        raise ValueError("val_perc must be between 0 and 100.")

    raw_roots = cfg.get('data.test.dataset_root')
    assert raw_roots is not None, "test.dataset_root is required"
    test_dataset_roots = split_csv_or_list(raw_roots)
    assert len(test_dataset_roots) >= 1, "test.dataset_root must not be empty"

    name_list = split_csv_or_list(cfg.get('data.test.dataset_name'))
    if len(test_dataset_roots) > 1:
        assert len(name_list) == len(test_dataset_roots), (
            "data.test.dataset_name must list one name per test set (comma-separated or YAML list), "
            f"got {len(name_list)} names for {len(test_dataset_roots)} dataset_root entries"
        )

    test_dataloaders = []
    for test_dataset_root in test_dataset_roots:
        test_data = ImageFolderDataset(
            dataset_root=test_dataset_root,
            image_dirname=cfg.get('data.test.image_dirname'),
            mask_dirname=cfg.get('data.test.mask_dirname'),
            depth_dirname=cfg.get('data.test.depth_dirname', None),
            transform=val_test_transform, list_name=None)
        test_dataloaders.append(torch.utils.data.DataLoader(
            test_data, batch_size=cfg.get('data.test.batch_size'), shuffle=False, num_workers=0))

    
    return train_dataloader, val_dataloader, test_dataloaders

from torch.utils.data import DataLoader
from data_provider.data_loader import (
    Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
)

DATA_MAP = {
    'ETTh1': (Dataset_ETT_hour, 'ETTh1.csv'),
    'ETTh2': (Dataset_ETT_hour, 'ETTh2.csv'),
    'ETTm1': (Dataset_ETT_minute, 'ETTm1.csv'),
    'ETTm2': (Dataset_ETT_minute, 'ETTm2.csv'),
    'Weather': (Dataset_Custom, 'weather.csv'),
    'Traffic': (Dataset_Custom, 'traffic.csv'),
    'Electricity': (Dataset_Custom, 'electricity.csv'),
    'Exchange': (Dataset_Custom, 'exchange_rate.csv'),
}


def data_provider(args, flag):
    dataset_cls, default_path = DATA_MAP[args.data]
    data_path = getattr(args, 'data_path', default_path) or default_path
    shuffle = (flag == 'train')

    dataset = dataset_cls(
        root_path=args.root_path,
        flag=flag,
        size=[args.seq_len, args.pred_len],
        features=args.features,
        data_path=data_path,
        target=args.target,
        scale=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    return dataset, loader

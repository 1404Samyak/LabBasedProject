import argparse
import os
import time
from scipy.io import loadmat, savemat
import numpy as np
import logging
import datetime

import torch
from torch import optim
from torch.utils.data import DataLoader

import network
import loaders


def main():
    start_time = time.time()

    # parse the input
    parser = argparse.ArgumentParser(description='DeepSIF Model')
    parser.add_argument('--save', type=int, default=True)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--arch', default='TemporalInverseNet', type=str)
    parser.add_argument('--dat', default='SpikeEEGBuild', type=str)
    parser.add_argument('--train', default='test_sample_source2.mat', type=str)
    parser.add_argument('--test', default='test_sample_source2.mat', type=str)
    parser.add_argument('--model_id', default=75, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--resume', default='', type=str) # Set to empty if changing architecture
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--fwd', default='leadfield_75_20k.mat', type=str)
    parser.add_argument('--rnn_layer', default=3, type=int)
    parser.add_argument('--info', default='', type=str)
    args = parser.parse_args()

    # ======================= PREPARE PARAMETERS =======================
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data_root = 'source/Simulation/'
    result_root = f'model_result/{args.model_id}_the_model'
    os.makedirs(result_root, exist_ok=True)

    # Load Forward Matrix to get dimensions
    fwd_data = loadmat(f'anatomy/{args.fwd}')['fwd']
    num_sensors = fwd_data.shape[0]
    num_regions = fwd_data.shape[1]

    # ======================= LOGGER =======================
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f'{result_root}/outputs_{args.arch}.log')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info(f"============================= {datetime.datetime.now()} =============================")
    logger.info(f"Training data: {args.train}, Testing data: {args.test}")
    logger.info(f"Detected Dimensions: Sensors={num_sensors}, Regions={num_regions}")

    for v in args.__dict__:
        if v not in ['workers', 'train', 'test']:
            logger.info(f'{v} = {args.__dict__[v]}')

    # ======================= LOAD DATA =======================
    # Passing fwd_data to loaders to ensure target alignment
    train_data = loaders.__dict__[args.dat](
        data_root + args.train,
        fwd=fwd_data,
        args_params={'dataset_len': 4}
    )

    test_data = loaders.__dict__[args.dat](
        data_root + args.test,
        fwd=fwd_data,
        args_params={'dataset_len': 4}
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True if torch.cuda.is_available() else False,
        shuffle=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True if torch.cuda.is_available() else False,
        shuffle=False
    )

    # ======================= CREATE MODEL =======================
    # FIXED: num_source now dynamically matches the fwd matrix (994)
    net = network.__dict__[args.arch](
        num_sensor=num_sensors,
        num_source=num_regions,
        rnn_layer=args.rnn_layer,
        spatial_model=network.MLPSpatialFilter,
        temporal_model=network.TemporalFilter,
        spatial_output='value_activation',
        temporal_output='rnn',
        spatial_activation='ELU',
        temporal_activation='ELU',
        temporal_input_size=500
    ).to(device)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-6)
    criterion = torch.nn.MSELoss(reduction='sum')

    args.start_epoch = 0
    best_result = np.inf
    train_loss = []
    test_loss = []

    # ======================= RESUME =======================
    if args.resume:
        ckpt_path = os.path.join(result_root, f'epoch_{args.resume}')
        print(f"=> Trying to load checkpoint {ckpt_path}")

        if os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            
            # Architecture check: Resume only if dimensions match
            try:
                net.load_state_dict(checkpoint['state_dict'], strict=True)
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch']
                best_result = checkpoint['best_result']
                print(f"=> Loaded checkpoint at epoch {args.start_epoch}")
            except RuntimeError as e:
                print(f"=> Resuming failed due to dimension mismatch: {e}")
                print("=> Starting fresh training with new dimensions.")
        else:
            print("=> No checkpoint found, starting fresh")

    print('Number of parameters:', net.count_parameters())
    print('Prepare time:', time.time() - start_time)

    # ======================= TRAINING =======================
    for epoch in range(args.start_epoch + 1, args.epoch + 1):

        train_lss_all = train(train_loader, net, criterion, optimizer, device)
        test_lss_all = validate(test_loader, net, criterion, device)

        train_loss.append(np.sum(train_lss_all) / len(train_data))
        test_loss.append(np.sum(test_lss_all) / len(test_data))

        msg = (
            f"Epoch {epoch} | "
            f"Train Loss {train_loss[-1]:.6f} | "
            f"Test Loss {test_loss[-1]:.6f}"
        )

        print(msg)
        logger.info(msg)

        is_best = test_loss[-1] < best_result
        best_result = min(best_result, test_loss[-1])

        if is_best:
            save_path = f'{result_root}/model_best.pth.tar'
            torch.save({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'best_result': best_result,
                'optimizer': optimizer.state_dict(),
                'attribute_list': getattr(net, 'attribute_list', [])
            }, save_path)

        if args.save:
            torch.save({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'best_result': best_result,
                'optimizer': optimizer.state_dict(),
                'attribute_list': getattr(net, 'attribute_list', [])
            }, f'{result_root}/epoch_{epoch}')

            savemat(
                f'{result_root}/train_test_error.mat',
                {'train_loss': train_loss, 'test_loss': test_loss}
            )


# ======================= TRAIN =======================
def train(train_loader, model, criterion, optimizer, device):
    model.train()
    losses = []

    for batch_idx, batch in enumerate(train_loader):
        data = batch['data'].to(device)
        nmm = batch['nmm'].to(device)

        optimizer.zero_grad()
        # Ensure we are comparing [Batch, Time, Sources]
        out = model(data)['last']
        
        loss = criterion(out, nmm)
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().item())

    return np.array(losses)


# ======================= VALIDATE =======================
def validate(loader, model, criterion, device):
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in loader:
            data = batch['data'].to(device)
            nmm = batch['nmm'].to(device)
            out = model(data)['last']
            loss = criterion(out, nmm)
            losses.append(loss.detach().cpu().item())

    return np.array(losses)


if __name__ == '__main__':
    main()
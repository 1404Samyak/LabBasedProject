import argparse
import os
import time
from scipy.io import loadmat, savemat
import numpy as np
import logging
import datetime
import collections

import torch
from torch.utils.data import DataLoader

import network
import loaders
from utils import get_otsu_regions


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description='DeepSIF Model')
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--dat', default='SpikeEEGBuildEval', type=str)
    parser.add_argument('--test', default='test_sample_source2.mat', type=str)
    parser.add_argument('--model_id', type=int, default=75)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--fwd', default='leadfield_75_20k.mat', type=str)
    parser.add_argument('--info', default='', type=str)
    parser.add_argument('--snr_rsn_ratio', default=0, type=float)
    parser.add_argument('--lfreq', default=-1, type=int)
    parser.add_argument('--hfreq', default=-1, type=int)
    args = parser.parse_args()

    # ======================= SETUP =======================
    device = torch.device("cpu")

    data_root = 'source/Simulation/'
    loadmat('anatomy/dis_matrix_fs_20k.mat')['raw_dis_matrix']

    result_root = f'model_result/{args.model_id}_the_model'
    if not os.path.exists(result_root):
        print(f"ERROR: No model {args.model_id}")
        return

    fwd = loadmat(f'anatomy/{args.fwd}')['fwd']

    # ======================= DATA =======================
    test_data = loaders.__dict__[args.dat](
        data_root + args.test,
        fwd=fwd,
        args_params={
            'snr_rsn_ratio': args.snr_rsn_ratio,
            'lfreq': args.lfreq,
            'hfreq': args.hfreq
        }
    )

    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=False
    )

    # ======================= MODEL =======================
    fn = os.path.join(
        result_root,
        'epoch_' + args.resume if args.resume else 'model_best.pth.tar'
    )

    print("=> Load checkpoint", fn)

    checkpoint = torch.load(
        fn,
        map_location='cpu',
        weights_only=False
    )

    net = network.__dict__[checkpoint['arch']](
        *checkpoint['attribute_list']
    ).to(device)

    net.load_state_dict(checkpoint['state_dict'], strict=False)
    net.eval()

    print('Number of parameters:', net.count_parameters())
    print('Prepare time:', time.time() - start_time)

    # ======================= EVAL =======================
    eval_dict = collections.defaultdict(list)

    with torch.no_grad():
        for batch_idx, sample_batch in enumerate(test_loader):
            if batch_idx > 0:
                break

            data = sample_batch['data'].to(device, torch.float)
            nmm = sample_batch['nmm'].numpy()
            label = sample_batch['label'].numpy()

            out = net(data)['last']
            eval_results = get_otsu_regions(out.cpu().numpy(), label)

            eval_dict['all_regions'].extend(eval_results['all_regions'])
            eval_dict['all_out'].extend(eval_results['all_out'])

            for kk in range(out.size(0)):

                # ✅ SAFE LABEL HANDLING
                if label.ndim == 3:
                    region_mask = label[kk, :, 0]
                else:
                    region_mask = label[kk]

                eval_dict['all_nmm'].append(
                    nmm[kk, :, region_mask]
                )

    savemat(
        fn + f"_preds_{args.test[:-4]}{args.info}.mat",
        eval_dict
    )


if __name__ == '__main__':
    main()

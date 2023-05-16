import random
import torch
from pathlib import Path
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor

import ipdb

class SintelDataset(data.Dataset):
    """Sintel recurrent dataset for training.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_REDS_GT.txt

    Each line contains:
    1. subfolder (clip) name; 2. frame number; 3. image shape, seperated by
    a white space.
    Examples:
    000 100 (720,1280,3)
    001 100 (720,1280,3)
    ...

    Key examples: "000/00000000"
    GT (gt): Ground-Truth, length is equal to LQ;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            val_partition (str): Validation partition types. 'REDS4' or
                'official'.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            interval_list (list): Interval list for temporal augmentation.
            random_reverse (bool): Random reverse input frames.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(SintelDataset, self).__init__()
        self.opt = opt
        
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(
            opt['dataroot_lq'])
        self.is_train = opt.get('is_train', False)
        self.is_val = opt.get('is_val', False)
        self.is_test = opt.get('is_test', False)
        
        self.num_frame = opt['num_frame']
        self.num_half_frames = opt['num_frame'] // 2
        self.keys = []

        with open(opt['meta_info_file'], 'r') as fin:
                for line in fin:
                    folder= line.split()[0]
                    self.keys.extend(
                        [f'{folder}/{i:06d}' for i in range(self.num_frame)])
        
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        

        # temporal augmentation configs
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        interval_str = ','.join(str(x) for x in opt['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt.get('gt_size', None)
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000
        center_frame_idx = int(frame_name)

        # determine the frameing frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = center_frame_idx - self.num_half_frames * interval
        end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval
        # each clip has 100 frames starting from 0 to 14
        while (start_frame_idx < 0) or (end_frame_idx > 14):
            center_frame_idx = random.randint(
                                    self.num_half_frames * interval, 
                                    14 - self.num_half_frames *interval)
            start_frame_idx = (center_frame_idx - self.num_half_frames * interval)
            end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval
        frame_name = f'{center_frame_idx:06d}'
        frame_list = list(
            range(start_frame_idx, self.num_frame, interval))
        # random reverse
        if self.random_reverse and random.random() < 0.5:
            frame_list.reverse()

        assert len(frame_list) == self.num_frame, (
            f'Wrong length of frame list: {len(frame_list)}')

        # get the GT frame (as the center frame)
        img_gts = []
        for frame in frame_list:
            img_gt_path = self.gt_root / clip_name / f'{frame:06d}.png'
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        # get the LQ frames
        img_lqs = []
        for frame in frame_list:
            img_lq_path = self.lq_root / clip_name / f'{frame:06d}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

        # randomly crop
        if self.is_train:
            img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size,
                                                  scale, clip_name)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        if self.is_train:
            img_lqs = augment(img_lqs, self.opt['use_flip'],
                              self.opt['use_rot'])

        img_results = img2tensor(img_lqs)
        img_lqs = torch.stack(img_results[:self.num_frame], dim=0)
        img_gts = torch.stack(img_results[self.num_frame:], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key, 'frame_list': frame_list}
        

    def __len__(self):
        return len(self.keys)

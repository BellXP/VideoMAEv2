import os
import torch
import argparse
import numpy as np

from timm.models import create_model

# NOTE: Do not comment `import models`, it is used to register models
import models
import utils
from dataset import video_transforms, volume_transforms
from dataset.loader import get_image_loader, get_skeleton_image_loader


def get_args():
    parser = argparse.ArgumentParser('', add_help=False)

    parser.add_argument('--model', default='vit_base_patch16_224', type=str)
    parser.add_argument('--tubelet_size', type=int, default=2)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--nb_classes', default=3, type=int)
    parser.add_argument('--num_frames', type=int, default=16)

    parser.add_argument('--resume', default='', help='checkpoint path')
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.add_argument('--normed_depth', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--device', default='cuda')

    parser.add_argument('--input_path', type=str)
    parser.add_argument('--fname_tmpl', default='{:05}.png', type=str)

    known_args, _ = parser.parse_known_args()
    return known_args


def main(args):
    device = torch.device(args.device)
    model = create_model(
        args.model,
        img_size=args.input_size,
        pretrained=False,
        num_classes=args.nb_classes,
        all_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        use_mean_pooling=args.use_mean_pooling
    )

    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // args.tubelet_size,
                        args.input_size // patch_size[0],
                        args.input_size // patch_size[1])
    args.patch_size = patch_size

    checkpoint = torch.load(args.resume, map_location='cpu')['model']
    print("Load ckpt from %s" % args.resume)

    # interpolate position embedding
    if 'pos_embed' in checkpoint:
        pos_embed_checkpoint = checkpoint['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]  # channel dim
        num_patches = model.patch_embed.num_patches  #
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches  # 0/1

        # height (== width) for the checkpoint position embedding
        orig_size = int(
            ((pos_embed_checkpoint.shape[-2] - num_extra_tokens) //
                (args.num_frames // model.patch_embed.tubelet_size))**0.5)
        # height (== width) for the new position embedding
        new_size = int(
            (num_patches //
                (args.num_frames // model.patch_embed.tubelet_size))**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" %
                    (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # B, L, C -> BT, H, W, C -> BT, C, H, W
            pos_tokens = pos_tokens.reshape(
                -1, args.num_frames // model.patch_embed.tubelet_size,
                orig_size, orig_size, embedding_size)
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size,
                                            embedding_size).permute(
                                                0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode='bicubic',
                align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
                -1, args.num_frames // model.patch_embed.tubelet_size,
                new_size, new_size, embedding_size)
            pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint['pos_embed'] = new_pos_embed
    elif args.input_size != 224:
        pos_tokens = model.pos_embed
        org_num_frames = 16
        T = org_num_frames // args.tubelet_size
        P = int((pos_tokens.shape[1] // T)**0.5)
        C = pos_tokens.shape[2]
        new_P = args.input_size // patch_size[0]
        # B, L, C -> BT, H, W, C -> BT, C, H, W
        pos_tokens = pos_tokens.reshape(-1, T, P, P, C)
        pos_tokens = pos_tokens.reshape(-1, P, P, C).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens,
            size=(new_P, new_P),
            mode='bicubic',
            align_corners=False)
        # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
        pos_tokens = pos_tokens.permute(0, 2, 3,
                                        1).reshape(-1, T, new_P, new_P, C)
        pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
        model.pos_embed = pos_tokens  # update
    if args.num_frames != 16:
        org_num_frames = 16
        T = org_num_frames // args.tubelet_size
        pos_tokens = model.pos_embed
        new_T = args.num_frames // args.tubelet_size
        P = int((pos_tokens.shape[1] // T)**0.5)
        C = pos_tokens.shape[2]
        pos_tokens = pos_tokens.reshape(-1, T, P, P, C)
        pos_tokens = pos_tokens.permute(0, 2, 3, 4,
                                        1).reshape(-1, C, T)  # BHW,C,T
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=new_T, mode='linear')
        pos_tokens = pos_tokens.reshape(1, P, P, C,
                                        new_T).permute(0, 4, 1, 2, 3)
        pos_tokens = pos_tokens.flatten(1, 3)
        model.pos_embed = pos_tokens  # update

    utils.load_state_dict(model, checkpoint)
    model.to(device)


    # sample preprocess
    image_loader = get_skeleton_image_loader() if args.normed_depth else get_image_loader()
    data_transform = video_transforms.Compose([
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    assert len(os.listdir(args.input_path)) == 300
    tick = 300 / float(args.num_frames)
    sample_indexes = [int(tick * x) for x in range(args.num_frames)]
    imgs = []
    for idx in sample_indexes:
        frame_fname = os.path.join(args.input_path, args.fname_tmpl.format(idx))
        imgs.append(image_loader(frame_fname, (args.input_size, args.input_size)))
    imgs = np.array(imgs)
    eval_sample = data_transform(imgs).unsqueeze(0).to(device)

    model_output = model(eval_sample)
    class_idx = model_output.max(1)[1].item()
    class_dict = {0: 'Neg', 1: 'Cri', 2: 'Pos'}
    print(f"The class of {args.input_path} is {class_dict[class_idx]}")


if __name__ == '__main__':
    opts = get_args()
    main(opts)

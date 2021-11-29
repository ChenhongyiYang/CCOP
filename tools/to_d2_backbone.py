import os
import argparse
import torch




def convert_resnet_soco_moco_d2_version(in_ckpt, out_file):
    ckpt = torch.load(in_ckpt, map_location=torch.device('cpu'))['state_dict']
    det_model = dict()
    det_model['model'] = dict()

    converted = []

    for k, v in ckpt.items():
        if k in converted or 'backbone_q' not in k or 'num_batches_tracked' in k:
            continue
        det_model['model'][k.replace('backbone_q', 'backbone')] = v
        converted.append(k)

    torch.save(det_model, out_file)
    print('Finished! Detection model saved to %s'%out_file)

    cls_model = dict()
    cls_model['model'] = dict()
    converted = []

    for k, v in ckpt.items():
        if k in converted or 'backbone_q' not in k or 'num_batches_tracked' in k or 'fpn' in k:
            continue
        cls_model['model'][k.replace('backbone_q.bottom_up.', '')] = v
        converted.append(k)

    torch.save(cls_model, out_file.replace('.pth', '_cls.pth'))
    print('Finished! FPN detection model saved to %s' % out_file.replace('.pth', '_cls.pth'))

    c5_det_model = dict()
    c5_det_model['model'] = dict()

    converted = []
    for k, v in ckpt.items():
        if k in converted or 'backbone_q' not in k or 'num_batches_tracked' in k or 'fpn' in k:
            continue
        if not 'res5' in k:
            c5_det_model['model'][k.replace('backbone_q.bottom_up.', 'backbone.')] = v
        else:
            c5_det_model['model'][k.replace('backbone_q.bottom_up.', 'roi_heads.')] = v
        converted.append(k)

    # print(c5_det_model['model'].keys())

    torch.save(c5_det_model, out_file.replace('.pth', '_c5.pth'))
    print('Finished! C5 detection model saved to %s' % out_file.replace('.pth', '_c5.pth'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model Conversion")
    parser.add_argument("--input", required=True, help="input model")
    parser.add_argument("--output", required=True, help="output model")
    args = parser.parse_args()

    if not os.path.isdir(os.path.split(args.output)[0]):
        os.mkdir(os.path.split(args.output)[0])

    convert_resnet_soco_moco_d2_version(args.input, args.output)


from pathlib import Path
import csv
import argparse

parser = argparse.ArgumentParser(description='create csv file for inference')
parser.add_argument('left_dir', help='path to dir containing left frames')
parser.add_argument('right_dir', help='path to dir containing right frames')
parser.add_argument('--out_path', help='',default='./paths.csv')
parser.add_argument('--disparity_out_dir', help='dir to save output disparities')
parser.add_argument('--segmentation_out_dir', help='dir to save output segmentation')



if __name__ == '__main__':
    args = parser.parse_args()
    left_dir = Path(args.left_dir)
    right_dir = Path(args.right_dir)
    assert left_dir.is_dir() and right_dir.is_dir()
    paths =dict()

    paths['left'] = sorted([p.resolve() for p in left_dir.glob('*.png')])
    # ensure that whatever is in the left dir is also in the right
    paths['right'] = sorted([p.resolve() for p in right_dir.glob('*.png')])

    for l_p, r_p in zip(paths['left'], paths['right']):
        assert l_p.name == r_p.name

    if args.disparity_out_dir is not None:
        paths['disp'] = [Path(args.disparity_out_dir).resolve()/p.name for p in paths['left']]
    else:
        paths['disp'] = [None]*len(paths['left'])


    if args.disparity_out_dir is not None:
        paths['segmentation'] = [Path(args.segmentation_out_dir).resolve()/p.name for p in paths['left']]
    else:
        paths['segmentation'] = [None]*len(paths['left'])

    # white csv
    with open(args.out_path, 'w', newline='') as csvfile:
        fieldnames = ['left', 'right', 'segmentation', 'disparity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for left_p, right_p, seg_p, disp_p in zip(*paths.values()):
            writer.writerow({'left':left_p, 'right':right_p, 'segmentation':seg_p, 'disparity':disp_p})



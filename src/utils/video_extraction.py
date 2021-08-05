import argparse
import os
from pathlib import Path

import cv2
from tqdm import tqdm

from image_processing import crop_square

image_size = 224


def extract_every_n_frames(src, dest=None, n=1, crop=True, inputShape=(224, 224)):
    """
    takes a folder of video files (mov or mp4) and extracts ever n'th frame. Images are either cropped or scaled to fit the inputshape
    :param src: path to a folder containing video files
    :param dest: folder to store the images. default is same as src. will be created, if not existing.
    :param n: every n'th frame will be extracted. default is 1 (every single frame)
    :param crop: True: take the biggest square region  of the Video as image(and scale it to inputshape) ; False Scale the video to inputShape
    :param inputShape: dimenstions of the resulting images as tuple (x,y). should match the supposed Prediction model. default is (224,224) for Mobile_net_v2
    :return:
    """
    if dest is None:
        dest = src
    os.makedirs(dest, exist_ok=True)
    src_p = Path(src)
    dest_p = Path(dest)
    file_extensions = ('**/*.MOV', '**/*.mp4')
    video_file_list = []
    for ext in file_extensions:
        video_file_list.extend(src_p.glob(ext))

    for video_file in tqdm(video_file_list):
        relative_video_p = video_file.relative_to(src_p)
        dest_for_video = dest_p.joinpath(relative_video_p.parent)
        dest_for_video.mkdir(parents=True, exist_ok=True)
        extract_every_n_frames_singe_file(video_file, dest_for_video, n=n, crop=crop, inputShape=inputShape)

    # for path, subdirs, files in os.walk(src):
    #     vid_number = 0
    #     for vid in files:
    #         video_name = vid[:vid.index('.')]
    #         if (vid.endswith('.mp4') or vid.endswith('.MOV')) and not vid.startswith('.'):
    #             print('found Video ', vid)
    #             save_dir = os.path.join(dest, os.path.basename(path))
    #             # save_dir = os.path.join(save_dir_root, 'data_set', os.path.basename(os.path.dirname(path)),
    #             #                         os.path.basename(path))
    #             print('going to save to: ', save_dir)
    #
    #             if not os.path.isdir(os.path.dirname(os.path.dirname(save_dir))):
    #                 os.makedirs(os.path.dirname(os.path.dirname(save_dir)))
    #                 print('created dir ', save_dir)
    #
    #             extract_every_n_frames_singe_file(os.path.join(path, vid), dest, n, crop, inputShape)
    #
    #             vid_number += 1


def extract_every_n_frames_singe_file(src:Path, dest:Path=None, n=1, crop=True, inputShape=(224, 224)):
    if not (src.name.endswith('.MOV') or src.name.endswith('.mp4')):
        print('not a video File! Only MOV and MP4 supported! Aborting... ', src)
        return

    if dest is None:
        dest = src
    dest.mkdir(parents=True,exist_ok=True)

    vid = os.path.basename(src)
    video_name = vid[:vid.index('.')]

    pbar = tqdm(desc=f'extract from {src.name}')

    vidcap = cv2.VideoCapture(str(src))
    frame = 0
    success, image = vidcap.read()
    while success:
        frame += 1
        if frame % n == 0:
            # print('read frame: ', frame)
            if str(image) == 'None':
                print('Image seems to be corrupted! Skipping to next')
                # success, image = vidcap.read()
            else:
                #image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                if crop:
                    image = crop_square(image)
                image = cv2.resize(image, inputShape)
                imgName = dest.joinpath(str(video_name + '_frame_' + str(frame) + '.jpg'))
                isWritten = cv2.imwrite(str(imgName), image)
                if not isWritten:
                    print(f'!---saving didnt work: {imgName}')
        pbar.update()
        success, image = vidcap.read()
    try:
        vidcap.release()
        cv2.destroyAllWindows()
    except:
        print('something went wrong. couldn\'t release vidcap')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", help="directory with videos to be processed. Or path to single Video file.",
                        required=True)
    parser.add_argument("--save_dir", help="directory to save the extracted images", required=True)
    parser.add_argument("--size", type=int,
                        help="size of the images after preprocessing. Images are processed as squares, so only one value is needed. Standard size = 224")
    parser.add_argument("--n", type=int, help="every n\'th frame will be extracted. Default is 1 (every single frame)")

    args = parser.parse_args()

    if args.size:
        image_size = args.size

    n = 1
    if args.n:
        n = args.n

    args.video_dir=Path(args.video_dir)
    args.save_dir=Path(args.save_dir)
    if args.video_dir.name.endswith('.MOV') or args.video_dir.name.endswith('.mp4'):
        extract_every_n_frames_singe_file(src=args.video_dir, dest=args.save_dir, n=n, crop=True,
                                          inputShape=(image_size, image_size))
    else:
        extract_every_n_frames(src=args.video_dir, dest=args.save_dir, n=n, crop=True,
                               inputShape=(image_size, image_size))

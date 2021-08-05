import argparse

import numpy
import cv2
import scipy
from scipy import ndimage
from random import randint
import random
import os
import shutil
from PIL import Image
from PIL import ImageEnhance
from PIL import ExifTags
from PIL import ImageFilter


# crop a random sized (60-100% of original image) square somewhere on the image
def crop_random_square(img):
    y, x = img.shape[0], img.shape[1]
    crop_size = int(min([y, x]) * random.uniform(0.6, 1.0))
    startx = randint(0, x - crop_size)
    starty = randint(0, y - crop_size)
    return img[starty:starty + crop_size, startx:startx + crop_size]


# crop a random sized (60-100% of original image) square somewhere on the image using PIL (Python Imaging Library)
def crop_random_square_pil(img):
    width = img.size[0]
    height = img.size[1]
    crop_size = int(min([width, height]) * random.uniform(0.6, 1.0))
    startx = randint(0, width - crop_size)
    starty = randint(0, width - crop_size)
    cropped = img.crop(
        (
            startx,
            starty,
            startx + crop_size,
            starty + crop_size
        )
    )
    return cropped


# crop a random sized (30% of original image) square somewhere on the image
def crop_random_background_square(img):
    y, x = img.shape[0], img.shape[1]
    if (min(img.shape[0], img.shape[1]) > 227):
        crop_size = int(max([0.3 * min([y, x]), 227]))
    else:
        crop_size = min([img.shape[0], img.shape[1]])
    startx = randint(0, x - crop_size)
    starty = randint(0, y - crop_size)
    return img[starty:starty + crop_size, startx:startx + crop_size]


# crop a square in the center of the image. Square will be as big as it can be
def crop_square(img):
    y, x = img.shape[0], img.shape[1]
    crop_size = min([y, x])
    startx = x // 2 - (crop_size // 2)
    starty = y // 2 - (crop_size // 2)
    return img[starty:starty + crop_size, startx:startx + crop_size]


# crop a square in the center of the image using PIL (Python Imaging Library). Square will be as big as it can be
def crop_square_pil(img):
    width = img.size[0]
    height = img.size[1]
    crop_size = min([width, height])
    cropped = img.crop(
        (
            width / 2 - crop_size / 2,
            height / 2 - crop_size / 2,
            width / 2 + crop_size / 2,
            height / 2 + crop_size / 2
        )
    )
    return cropped

# crop a quarter from a square-sized image. Index
def crop_quarter_square_pil(img, index):
    width = img.size[0]
    height = img.size[1]
    i = index % 4
    if i < 2:
        crop_width_1 = 0
        crop_width_2 = width/2
    else:
        crop_width_1 = width/2
        crop_width_2 = width

    if i == 0 or i == 2:
        crop_height_1 = 0;
        crop_height_2 = height/2
    else:
        crop_height_1 = height/2
        crop_height_2 = height
    cropped = img.crop(
        (
            crop_width_1,
            crop_height_1,
            crop_width_2,
            crop_height_2
        )
    )
    return cropped



# rotate image randomly between -15 to 15 degrees
def rotate_radomly(img):
    angle = randint(-15, 15)
    return ndimage.rotate(img, angle, reshape=False)


# rotate image randomly between -15 to 15 degrees using PIL (Python Imaging Library)
def rotate_randomly_pil(img):
    return img.rotate(randint(-15, 15))


# preprocess training images (without adding background!) and use sobel operator for edge detection
# NOTE: no background will be added. 
def detect_train_sobel_edges_in_images(im_dir, sav_dir):
    for x in range(1, 2):
        print(x)
        image_string = im_dir + 'screenshot' + str(x) + '.png'
        save_string = sav_dir + 'preprocessed' + str(x) + '.jpg'
        im = scipy.misc.imread(image_string)
        resize_width, resize_height = int(0.5 * im.shape[0]), int(0.5 * im.shape[1])
        im = scipy.misc.imresize(im, [resize_width, resize_height])
        im = rotate_radomly(im)
        im = crop_square(im)
        im = crop_random_square(im)
        im = scipy.misc.imresize(im, [112, 112])
        im = im.astype('int32')
        dx = ndimage.sobel(im, 0)  # horizontal derivative
        dy = ndimage.sobel(im, 1)  # vertical derivative
        mag = numpy.hypot(dx, dy)  # magnitude
        mag *= 255.0 / numpy.max(mag)  # normalize (Q&D)
        scipy.misc.imsave(save_string, mag)


# preprocess test images and use sobel operator for edge detection
def detect_test_sobel_edges_in_images():
    for x in range(1, 5506):
        image_string = 'zylinder_images/leo_images/Systembediengeraet_A110/merged' + str(x) + '.jpg'
        save_string = 'zylinder_images/nur_vorne_kanten/Systembediengeraet_A110/sobel' + str(x) + '.jpg'
        im = scipy.misc.imread(image_string)
        im = crop_square(im)
        im = scipy.misc.imresize(im, [112, 112])
        im = im.astype('int32')
        dx = ndimage.sobel(im, 0)  # horizontal derivative
        dy = ndimage.sobel(im, 1)  # vertical derivative
        mag = numpy.hypot(dx, dy)  # magnitude
        mag *= 255.0 / numpy.max(mag)  # normalize (Q&D)
        scipy.misc.imsave(save_string, mag)

        if x % 55 == 0:
            print('\r' + str(int(x / 55)) + '% done', end='')
            # os.remove(image_string)


# do a preprocessing for image without doing any edge detection and without adding background to the image
# NOTE: No background will be added to the image
def preprocess_images_without_sobel(im_dir, sav_dir):
    for x in range(1, 536):
        image_string = im_dir + 'screenshot' + str(x) + '.png'
        im = scipy.misc.imread(image_string)
        resize_width, resize_height = int(0.5 * im.shape[0]), int(
            0.5 * im.shape[1])  # reduce image_size to 0.5 * image_size for faster processing
        im = scipy.misc.imresize(im, [resize_width, resize_height])  # reduce image_size for faster processing

        for y in range(1, 11):
            save_string = sav_dir + 'preprocessed' + str(x) + '_' + str(y) + '.jpg'
            image = rotate_radomly(im)
            image = crop_square(image)
            image = crop_random_square(image)
            image = scipy.misc.imresize(image, [112, 112])
            image = image.astype('int32')
            scipy.misc.imsave(save_string, image)
        print('\rpreprocessing:' + str(round(x / 5.34, 2)) + '% completed', end='')


# do a preprocessing for image without doing any edge detection
def preprocess_background_images(im_dir, sav_dir):
    for x in range(1, 2469):
        for y in range(1, 10):
            image_string = im_dir + 'background' + str(x) + '.jpg'
            save_string = sav_dir + 'preprocessed' + str(x) + '_' + str(y) + '.jpg'
            im = scipy.misc.imread(image_string)
            # resize_width, resize_height = int(0.5 * im.shape[0]), int(
            #    0.5 * im.shape[1])  # reduce image_size to 0.5 * image_size for faster processing
            # im = scipy.misc.imresize(im,
            #                         [resize_width, resize_height])  # reduce image_size for faster processing

            im = crop_random_background_square(im)
            im = scipy.misc.imresize(im, [299, 299])
            im = im.astype('int32')

            scipy.misc.imsave(save_string, im)
        if (x % 10 == 0 or x % 24 == 0):
            print('\rpreprocessing:' + str(round(x / 24.68, 2)) + '% completed', end='')


# edge detection with prewitt filter
def detect_prewitt_edges_in_images():
    for x in range(303, 310):
        print(x)
        image_string = 'zylinder_images/5045/IMG_0' + str(x) + '.png'
        save_string = 'Prewitt_Bilder/prewitt' + str(x) + '.jpg'
        im = scipy.misc.imread(image_string)
        im = im.astype('int32')
        dx = ndimage.prewitt(im, 0)  # horizontal derivative
        dy = ndimage.prewitt(im, 1)  # vertical derivative
        mag = numpy.hypot(dx, dy)  # magnitude
        mag *= 255.0 / numpy.max(mag)  # normalize (Q&D)
        scipy.misc.imsave(save_string, mag)


# edge detection with canny filter
def detect_canny_edges_in_images():
    for x in range(1, 30):
        if x % 55 == 0:
            print('\r' + str(int(x / 55)) + '% done', end='')
        image_string = 'zylinder_images/nur_vorne_kanten/preprocessed/betriebsstufenbediengeraet_A4/test/test_resized' + str(
            x) + '.jpg'
        save_string = 'zylinder_images/nur_vorne_kanten/preprocessed/betriebsstufenbediengeraet_A4/test/canny' + str(
            x) + '.jpg'
        img = cv2.imread(image_string, 0)

        v = numpy.median(img)

        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(img, lower, upper)

        cv2.imwrite(save_string, edges)
        os.remove(image_string)


# resize images to a size of 112x112 so they can be used for the neural network
def resize():
    for x in range(1, 100):
        print(x)
        image_string = 'zylinder_images/zylinder/preprocessed/druckspeicher/test/test' + str(x) + '.jpg'
        save_string = 'zylinder_images/zylinder/preprocessed/druckspeicher/test/test' + str(x) + '.jpg'

        im = scipy.misc.imread(image_string)
        im = crop_square(im)
        im = scipy.misc.imresize(im, [112, 112])
        scipy.misc.imsave(save_string, im)
        # os.remove(image_string)


# crop images to a square and then resize them to 112x112
def reshape_and_resize():
    for x in range(1, 364):
        print(str(x))
        image_string = '/Users/adminsitrator/denkbares/deep_learning_stuff/bilder_deep_learning/zylinder/Hydraulik_Druckspeicher/screenshot' + str(
            x) + '.png'
        save_string = '/Users/adminsitrator/denkbares/deep_learning_stuff/bilder_deep_learning/zylinder_images/zylinder/unprocessed_299/Hydraulikdruckspeicher/screenshot' + str(
            x) + '.jpg'
        im = scipy.misc.imread(image_string)
        im = crop_square(im)
        im = scipy.misc.imresize(im, [299, 299])
        scipy.misc.imsave(save_string, im)
        if (x % 10 == 0):
            print('\rpreprocessing:' + str(round(x / 20.57, 2)) + '% completed', end='')


# us a video as input and extract single frames as images
def extract_images_from_video(video_path, save_dir, width, height):
    vidcap = cv2.VideoCapture(video_path)
    #success, image = vidcap.read()
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if not image is None:
            # image = cv2.flip(image, -1)
            image = cv2.resize(image, (width, height))
            print('Read a new frame %d: ' % count, success)
            cv2.imwrite(save_dir + "frame%d.jpg" % count, image)  # save frame as JPEG file
            count += 1


# remove grey background from training images and make it transparent
def scrap_background(img, save):
    img = Image.open(img)
    img = img.convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:  # if it's grey like the background -> make it transparent
        if (210 <= item[0] <= 250 and 210 <= item[1] <= 250 and 210 <= item[2] <= 250):
            # if (227 <= item[0] <= 231 and 227 <= item[1] <= 231 and 227 <= item[2] <= 231):
            newData.append((item[0], item[1], item[2], 0))  # set alpha-value 0
        else:
            newData.append(item)

    img.putdata(newData)
    img.save(save, "PNG")


# merge two images to images. use one image as background and the other one will be in the foreground
def merge_images(im1, im2, im_size):
    background = Image.open(im1).resize([im_size, im_size])
    foreground = Image.open(im2).resize([im_size, im_size])
    background.paste(foreground, (0, 0), foreground)
    return background

def merge_image_with_random_noise(image, im_size):
    random_noise = numpy.dstack((255 * numpy.random.random((im_size, im_size)), 255 * numpy.random.random((im_size, im_size)), 255 * numpy.random.random((im_size, im_size))))
    background = Image.fromarray(numpy.uint8(random_noise))
    foreground = Image.open(image).resize([im_size, im_size])
    background.paste(foreground, (0, 0), foreground)
    return background

# merge a batch of images with background images
def create_merged_images():
    for i in range(1, 5351):
        for j in range(0, 5):
            im1 = 'clutter_backgrounds/preprocessed/preprocessed' + str(i + j * 3000) + '.jpg'
            im2 = 'zylinder_images/leo_images_nur_vorne/preprocessed/Systembediengeraet_A110/transparent' + str(
                i) + '.jpg'
            sav = 'zylinder_images/leo_images/Systembediengeraet_A110/merged' + str(i) + '_' + str(j) + '.jpg'
            merge_images(im1, im2, sav)

        if i % 53 == 0:
            print('\r' + str(int(i / 53.4)) + '% done', end='')


# horizontally flip images
def flip_images():
    for x in range(1, 32101):
        image_string = 'zylinder_images/leo_images/Systembediengeraet_A110/preprocessed' + str(x) + '.jpg'
        save_string = 'zylinder_images/leo_images/Systembediengeraet_A110/preprocessed' + str(32100 + x) + '.jpg'
        im = scipy.misc.imread(image_string)
        horizontal_im = cv2.flip(im, 0)
        scipy.misc.imsave(save_string, horizontal_im)
        if x % 32 == 0:
            print('\r' + str(int(x / 320)) + '% done', end='')


def change_brightness(image, save):
    # manipulate brightness of the image
    brightness = ImageEnhance.Brightness(image)
    brightness_manipulated = brightness.enhance(random.uniform(0.6, 1.5))
    rgb_im = brightness_manipulated.convert('RGB')
    rgb_im.save(save, "JPEG")


# manipulate HSV-channels of the whole image
def manipulate_hsv(image):
    im = image.convert('HSV')
    im_arr = numpy.array(im)
    h_vals = random.uniform(0.7, 1.6) * (im_arr[..., 0])
    s_vals = random.uniform(0.3, 2.6) * (im_arr[..., 1] + randint(1,
                                                                  30))  # components have lots of grey colors -> grey means saturation == 0 -> give a little more saturation, so that manipulation is successful
    v_vals = random.uniform(0.7, 1.6) * im_arr[..., 2]

    # S and V channels should not be greater than 255. H channel can be greater, because it starts from beginning and beginning is the continuous successor of the end -> see HSV cone
    s_vals[s_vals > 255] = 255
    v_vals[v_vals > 255] = 255

    im_arr[..., 0] = h_vals
    im_arr[..., 1] = s_vals
    im_arr[..., 2] = v_vals

    manipulated_image = Image.fromarray(im_arr, mode='HSV')
    return manipulated_image.convert('RGB')


# manipulate HSV-channels of the whole image variante 2
def manipulate_hsv_addition(image):
    im = image.convert('HSV')
    im_arr = numpy.array(im, dtype=numpy.uint16)

    h_vals = im_arr[..., 0]
    s_vals = im_arr[..., 1]
    v_vals = im_arr[..., 2]

    h_vals = h_vals + randint(-20, 20)
    s_vals = s_vals + randint(-40, 40)
    v_vals = v_vals + randint(-40, 40)

    s_vals[s_vals < 0] = 0
    s_vals[s_vals > 255] = 255
    v_vals[v_vals < 0] = 0
    v_vals[v_vals > 255] = 255

    im_arr[..., 0] = h_vals
    im_arr[..., 1] = s_vals
    im_arr[..., 2] = v_vals

    im_arr = numpy.array(im_arr, dtype=numpy.uint8)  # Pillow needs an 8bit array to form a picture from the array
    manipulated_image = Image.fromarray(im_arr, mode='HSV')
    # manipulated_image.show()

    return manipulated_image.convert('RGB')


# manipulate every single pixel's HSV-values
def manipulate_every_pixels_hsv(image):
    # image = Image.open(image)
    hsv_im = image.convert('HSV')
    im_arr = numpy.array(hsv_im)
    height, width, _ = im_arr.shape
    for j in range(width):
        for i in range(height):
            im_arr[i][j][0] = min(random.uniform(0.7, 1.6) * im_arr[i][j][0], 255)  # H-value
            im_arr[i][j][1] = min(random.uniform(0.7, 1.6) * im_arr[i][j][1], 255)  # S-value
            im_arr[i][j][2] = min(random.uniform(0.7, 1.6) * im_arr[i][j][2], 255)  # V-value

    manipulated_image = Image.fromarray(im_arr, mode='HSV')
    return manipulated_image.convert('RGB')


def manipulate_rgb(image):
    rgb_im = image.convert('RGB')
    im_arr = numpy.array(rgb_im, dtype=numpy.uint16)  # we need 16bit int, because 8bit only works until 255

    r_vals = im_arr[..., 0]
    g_vals = im_arr[..., 1]
    b_vals = im_arr[..., 2]

    r_vals = r_vals + randint(-20, 20)
    g_vals = g_vals + randint(-20, 20)
    b_vals = b_vals + randint(-20, 20)

    r_vals[r_vals < 0] = 0
    r_vals[r_vals > 255] = 255
    g_vals[g_vals < 0] = 0
    g_vals[g_vals > 255] = 255
    b_vals[b_vals < 0] = 0
    b_vals[b_vals > 255] = 255

    im_arr[..., 0] = r_vals
    im_arr[..., 1] = g_vals
    im_arr[..., 2] = b_vals

    im_arr = numpy.array(im_arr, dtype=numpy.uint8)  # Pillow needs an 8bit array to form a picture from the array
    im = Image.fromarray(im_arr, mode='RGB')
    im.show()


# equalize the histogram of the luminance (Y-channel) of an image
def equalize_luminance(image):
    pil_im = image.convert('RGB')
    img = numpy.array(pil_im)
    img = img[:, :, ::-1].copy()
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    cv2_im = img_output
    pil_im = Image.fromarray(cv2_im)
    # pil_im.show()
    return pil_im

def blur_images(im_dir):
    for root, dirs, files in os.walk(im_dir):
        for idx, file in enumerate(files):
            if file.endswith(".jpg"):
                try:
                    image_path = os.path.join(root, file)
                    print(image_path)
                    image = Image.open(image_path)
                    image = image.filter(ImageFilter.GaussianBlur(radius=1))
                    image.save(image_path, "JPEG")

                except Exception as e:
                    print(e)


# convert image from *.png to *.jpg
def convert_to_jpg(image):
    im = Image.open(image)
    # im = im.resize([224, 224])
    rgb_im = im.convert('RGB')
    rgb_im.save(image.replace('.png', '.jpg'))


def resizeImages(im_dir):
    for root, dirs, files in os.walk(im_dir):
        for idx, file in enumerate(files):
            if file.endswith(".JPG"):
                try:
                    image_path = os.path.join(root, file)
                    print(image_path)
                    image = Image.open(image_path)

                    for orientation in ExifTags.TAGS.keys():
                        if ExifTags.TAGS[orientation] == 'Orientation': break
                    exif = dict(image._getexif().items())
                    if exif[orientation] == 3:
                        image = image.rotate(180, expand=True)
                    elif exif[orientation] == 6:
                        image = image.rotate(270, expand=True)
                    elif exif[orientation] == 8:
                        image = image.rotate(90, expand=True)

                    basewidth = 1200
                    wpercent = (basewidth / float(image.size[0]))
                    hsize = int((float(image.size[1]) * float(wpercent)))
                    image.thumbnail((basewidth, hsize), Image.ANTIALIAS)
                    image.save(image_path, "JPEG")

                except Exception as e:
                    print(e)


# do the whole preprocessing process for one image
# preprocess image and merge it with several backgrounds
def preprocess_all(image_path, sav_dir, image_size, background_directory):
    sav_dir = sav_dir + os.path.basename(os.path.dirname(image_path))
    if not os.path.isdir(sav_dir):  # if directory doesn't exist, create it
        os.mkdir(sav_dir)
        # uri = os.path.dirname(image_path) + '/uri.txt'  // uncomment uri path
        # shutil.copyfile(uri, sav_dir + '/uri.txt')  // uncomment uri path


    # read image
    im = Image.open(image_path)
    width, height = im.size
    # im.resize([int(0.5 * width), int(0.5 * height)])  # resize image to 50% to accelerate computation

    for y in range(0, 6):
        save_string = sav_dir + "/" + os.path.basename(image_path)[:-4] + '_' + str(y) + '.jpg'
        image = rotate_randomly_pil(im)
        image = crop_square_pil(image)
        image = crop_random_square_pil(image)
        # image = crop_quarter_square_pil(image, y)
        image = image.resize([image_size, image_size])
        image.save(save_string, "PNG")

    for y in range(0, 6):
        foreground = sav_dir + "/" + os.path.basename(image_path)[:-4] + '_' + str(y) + '.jpg'
        for z in range(0, 1):
            background = background_directory + random.choice(
                os.listdir(background_directory))  # randomly choose a background image
            save_string_merged = sav_dir + "/" + os.path.basename(image_path)[:-4] + '_' + str(y) + '.jpg'
            merged_image = merge_images(background, foreground, image_size)
            # merged_image = merge_image_with_random_noise(foreground, image_size)
            # hsv_manipulated_image = manipulate_every_pixels_hsv(merged_image)
            hsv_manipulated_image = manipulate_hsv_addition(merged_image)
            # equalized_image = equalize_luminance(hsv_manipulated_image)
            change_brightness(hsv_manipulated_image, save_string_merged)


# initialize preprocessing for a directory.
# Directory should contain images for all classes in different directories
# structure should be as follows:
#
# im_dir  (<- the one you use as parameter)
# │
# │
# └───Class1
# │   │   image1.png
# │   │   image2.png
# │   │     ...
# │
# └───Class2
#     │   image1.png
#     │   image2.png
#     │     ...

def do_preprocessing_for_dir(im_dir, sav_dir, image_size, background_directory):
    if not os.path.exists(sav_dir):
        os.mkdir(sav_dir)
    for root, dirs, files in os.walk(im_dir):
        for idx, file in enumerate(files):
            if file.endswith(".png"):
                image_path = os.path.join(root, file)
                start_image_preprocessing(image_path, sav_dir, image_size, background_directory)
                printProgressBar(idx + 1, len(files), prefix='Progress:', suffix='Complete')
                # print('\rpreprocessing:' + str(round(idx / 4.84, 2)) + '% completed', end='')


# start the image preprocessing and take care of possibly occuring ValueErrors
def start_image_preprocessing(image_path, sav_dir, image_size, background_directory):
    try:
        preprocess_all(image_path, sav_dir, image_size, background_directory)
    except ValueError as e:  # quick and dirty Solution: sometimes a ValueError is raised while converting to HSV. Retrying always helps. So we catch it here and try again
        if str(e) == "conversion from L to HSV not supported":
            print(e)
            start_image_preprocessing(image_path, sav_dir, image_size, background_directory)
        else:
            raise


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='')
    # Print New Line on Complete
    if iteration == total:
        print()


if __name__ == '__main__':

    '''
    image_size = 299
    # background_directory = "/Users/adminsitrator/denkbares/deep_learning_stuff/bilder_deep_learning/clutter_backgrounds/preprocessed_299/"

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", help="image directory with images to be processed")
    parser.add_argument("--save_dir", help="directory to save the preprocessed images")
    parser.add_argument("--size", type=int,
                        help="size of the images after preprocessing. Images are processed as squares, so only one value is needed. Standard size = 299")
    parser.add_argument("--background_directory",
                        help="directory with background images. Background Images have to be same size as processed images. Optionally change parameter --size")
    args = parser.parse_args()

    if args.size:
        image_size = args.size
    if args.background_directory:
        background_directory = args.background_directory

    if args.image_dir and args.save_dir:
        im_dir = args.image_dir
        sav_dir = args.save_dir
        do_preprocessing_for_dir(im_dir, sav_dir, image_size, background_directory)


    else:
        print('Please specify image directory and output directory')
        
    '''




    i = 0
    for path, subdirs, files in os.walk('/Users/tobi/denkbares/deep_learning_stuff/Bilder/videos_AXION_test_16_klassen/'):
        for name in files:
            if name.endswith('.MOV'):
                i+=1
                print(os.path.basename(path))
                save_dir = '/Users/tobi/denkbares/deep_learning_stuff/Bilder/bilder_AXION_test_16_klassen/' + os.path.basename(path) + '/'
                if not os.path.isdir(os.path.dirname(os.path.dirname(save_dir))):
                    os.mkdir(os.path.dirname(os.path.dirname(save_dir)))
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                    i = 0
                extract_images_from_video(os.path.join(path, name), save_dir + str(i), 299, 299)
                




    # blur_images('/home/deep/workspace/deepbares/images/blurred_leo_can2019_02_11')



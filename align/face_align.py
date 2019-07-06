from PIL import Image
from align.detector import detect_faces
from align.align_trans import get_reference_facial_points, warp_and_crop_face
import numpy as np
import os
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "face alignment")
    parser.add_argument("-source_root", "--source_root", help = "specify your source dir", default = "/media/hyo/文档/VIS-NIR/CASIA_VIS_NIR/NIR/", type = str)
    parser.add_argument("-dest_root", "--dest_root", help = "specify your destination dir", default = "/media/hyo/文档/VIS-NIR/CASIA_VIS_NIR/NIR_Aligned/", type = str)
    parser.add_argument("-crop_size", "--crop_size", help = "specify size of aligned faces, align and crop with padding", default = 112, type = int)
    args = parser.parse_args()

    source_root = args.source_root # specify your source dir
    dest_root = args.dest_root # specify your destination dir
    crop_size = args.crop_size # specify size of aligned faces, align and crop with padding
    scale = crop_size / 112.
    reference = get_reference_facial_points(default_square = True) * scale

    # cwd = os.getcwd() # delete '.DS_Store' existed in the source_root
    # os.chdir(source_root)
    # os.system("find . -name '*.DS_Store' -type f -delete")
    # os.chdir(cwd)
    
    # source_root = 's3_NIR_20385_008.bmp'
    # source_root = 's1_VIS_00003_010.jpg'
    # target_root = 's3_NIR_20385_008.jpg'
    # img = Image.open(source_root)
    # img.save(target_root,'jpeg',quality=100)
    # img = Image.open(target_root)
    # img = img.convert('RGB')
    # arr_img = np.array(img)
    # # img = np.expand_dims(img, 0)
    # _, landmarks = detect_faces(img)
    # facial5points = [[landmarks[0][j], landmarks[0][j+5]] for j in range(5)]
    # warped_face = warp_and_crop_face(np.array(img),facial5points,reference,crop_size=(crop_size,crop_size))
    # img_warped = Image.fromarray(warped_face[0])
    # plt.imshow(img_warped)
    # plt.show()

    if not os.path.isdir(dest_root):
        os.mkdir(dest_root)


    for image_name in os.listdir(source_root):
        print("Processing\t{}".format(os.path.join(source_root, image_name)))
        img = Image.open(os.path.join(source_root, image_name))
        img.save(os.path.join(dest_root,image_name.split('.')[0]+'.jpg'),'jpeg',quality=100)
        img = Image.open(os.path.join(dest_root,image_name.split('.')[0]+'.jpg'))
        img = img.convert('RGB')
        arr_img = np.array(img)
        try: # Handle exception
            _, landmarks = detect_faces(img)
        except Exception:
            print("{} is discarded due to exception!".format(os.path.join(source_root, image_name)))
            continue
        if len(landmarks) == 0: # If the landmarks cannot be detected, the img will be discarded
            print("{} is discarded due to non-detected landmarks!".format(os.path.join(source_root, image_name)))
            continue
        facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
        warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
        img_warped = Image.fromarray(warped_face[0])
        if image_name.split('.')[-1].lower() not in ['jpg', 'jpeg']: #not from jpg
            image_name = '.'.join(image_name.split('.')[:-1]) + '.jpg'
        img_warped.save(os.path.join(dest_root, image_name.split('.')[0]+'.jpg'),quality=100)

import re
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from PIL import Image as im
import random
import pickle


# Counters for counting cases for each #ball
count0 = 0
count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
count6 = 0
count7 = 0

if __name__ == '__main__':
    path_save = 'C:/Users/user/PycharmProjects/tactile/patch_tactile_40mm/'
    image_path = "C:/Users/user/PycharmProjects/tactile/spread0/"

    file_id = glob.glob(image_path + '*/*/grasping_result.csv.', recursive=True)

    print(file_id[0:5])

    random.shuffle(file_id)
    print(file_id[0:5])

    with open(path_save + "file_id", "wb") as fp:
        pickle.dump(file_id, fp)

    print(len(file_id))

    # input('wait')

    data_train, data_val, data_test, data_test2 = [], [], [], []
    image_train, image_val, image_test, image_test2 = [], [], [], []

    data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7 = [], [], [], [], [], [], [], []
    image_0, image_1, image_2, image_3, image_4, image_5, image_6, image_7 = [], [], [], [], [], [], [], []

    counter = 0
    for file in file_id:
        print('counter', counter)
        counter += 1
        names = os.path.split(file)
        # print('names', names)
        file_1 = os.path.splitext(names[1])
        # print('file_1', file_1)
        base_name = re.split('(\d+)', file_1[0])
        base_name3 = base_name[:-1]
        base_name1 = base_name[:-3]
        # print('base_name', base_name)
        # print('base_name1', base_name1)
        base_name1 = ''.join(base_name1)
        base_name3 = ''.join(base_name3)
        # print('base_name1', base_name1)
        base_name2 = names[0] + '/' + base_name1
        base_name4 = names[0] + '/' + base_name3
        # print('base_name4', base_name4)
        # input('wait')
        src0 = base_name4 + 'back.jpg'
        src1 = base_name4 + 'front.jpg'
        src2 = base_name4 + 'left.jpg'
        src3 = base_name4 + 'right.jpg'
        src5 = base_name4 + 'sensor_reading.csv'

        dft = pd.read_csv(src5, header=None)

        data = dft.to_numpy()
        pdata = data[-5:, 1:25]
        pdata = np.mean(pdata, axis=0)
        for idx, value in enumerate(pdata):
            if value > -0.01:
                pdata[idx] = 0
        pdata = np.insert(pdata, 24, 0, axis=0)
        pdata = np.insert(pdata, 22, 0, axis=0)
        pdata = np.insert(pdata, 2, 0, axis=0)
        pdata = np.insert(pdata, 0, 0, axis=0)
        # pdata = normalize(pdata)
        # print(pdata)
        pdata.resize(7, 4)

        fdata0 = data[-5:, 25:59]
        fdata0 = np.mean(fdata0, axis=0)
        fdata0[24] = (fdata0[24] + fdata0[25]) / 2
        fdata0[25] = (fdata0[25] + fdata0[26]) / 2
        fdata0[26] = (fdata0[26] + fdata0[27]) / 2
        fdata0 = np.delete(fdata0, 27, 0)
        for idx, value in enumerate(fdata0):
            if value > -0.01:
                fdata0[idx] = 0
        # print(fdata0)
        # fdata0 = normalize(fdata0)
        fdata0.resize(11, 3)

        fdata1 = data[-5:, 125:159]
        fdata1 = np.mean(fdata1, axis=0)
        fdata1[24] = (fdata1[24] + fdata1[25]) / 2
        fdata1[25] = (fdata1[25] + fdata1[26]) / 2
        fdata1[26] = (fdata1[26] + fdata1[27]) / 2
        fdata1 = np.delete(fdata1, 27, 0)
        for idx, value in enumerate(fdata1):
            if value > -0.01:
                fdata1[idx] = 0
        # print(fdata1)
        # fdata1 = normalize(fdata1)
        fdata1.resize(11, 3)

        fdata2 = data[-5:, 225:259]
        fdata2 = np.mean(fdata2, axis=0)
        fdata2[24] = (fdata2[24] + fdata2[25]) / 2
        fdata2[25] = (fdata2[25] + fdata2[26]) / 2
        fdata2[26] = (fdata2[26] + fdata2[27]) / 2
        fdata2 = np.delete(fdata2, 27, 0)
        for idx, value in enumerate(fdata2):
            if value > -0.01:
                fdata2[idx] = 0
        # print(fdata2)
        # fdata2 = normalize(fdata2)
        fdata2.resize(11, 3)

        width = 32
        m_factor = 128

        data1 = -m_factor * pdata
        Image0 = im.fromarray(data1)
        ratio = (width / float(Image0.size[0]))
        hsize = int((float(Image0.size[1]) * float(ratio)))
        Image0 = Image0.resize((width, hsize), im.ANTIALIAS)
        # Image0.show()
        rgb_img0 = Image0.convert('RGB')
        # rgb_img.save("GT/pdata.jpg")

        data1 = -m_factor * fdata0
        Image1 = im.fromarray(data1)
        ratio = (width / float(Image1.size[0]))
        hsize = int((float(Image1.size[1]) * float(ratio)))
        Image1 = Image1.resize((width, hsize), im.ANTIALIAS)
        # Image1.show()
        rgb_img = Image1.convert('RGB')
        # rgb_img.save("GT/fdata0.jpg")

        X0a = np.concatenate((rgb_img0, rgb_img), axis=0)
        # plt.imshow(X0a / 255)
        # plt.show()

        data1 = -m_factor * fdata1
        Image1 = im.fromarray(data1)
        ratio = (width / float(Image1.size[0]))
        hsize = int((float(Image1.size[1]) * float(ratio)))
        Image1 = Image1.resize((width, hsize), im.ANTIALIAS)
        # Image1.show()
        rgb_img = Image1.convert('RGB')
        # rgb_img.save("GT/fdata1.jpg")

        X0a = np.concatenate((X0a, rgb_img), axis=0)
        # plt.imshow(X0a / 255)
        # plt.show()

        data1 = -m_factor * fdata2
        Image1 = im.fromarray(data1)
        ratio = (width / float(Image1.size[0]))
        hsize = int((float(Image1.size[1]) * float(ratio)))
        Image1 = Image1.resize((width, hsize), im.ANTIALIAS)
        # Image1.show()
        rgb_img = Image1.convert('RGB')
        # rgb_img.save("GT/fdata2.jpg")

        X0a = np.concatenate((X0a, rgb_img), axis=0)
        X0a = cv2.resize(X0a, (32, 256))
        # plt.imshow(X0a / 255)
        # plt.show()

        # print(X0a.shape)

        X0 = cv2.imread(src0)
        X1 = cv2.imread(src1)
        X2 = cv2.imread(src2)
        X3 = cv2.imread(src3)

        X0 = cv2.resize(X0, (128, 128))
        X1 = cv2.resize(X1, (128, 128))
        X2 = cv2.resize(X2, (128, 128))
        X3 = cv2.resize(X3, (128, 128))

        # remove the boxes in images 0-3
        image_z = np.zeros((128, 128, 3), dtype=np.uint8)
        X0[100:, :, :] = image_z[100:, :, :]
        X1[100:, :, :] = image_z[100:, :, :]
        X2[100:, :, :] = image_z[100:, :, :]
        X3[100:, :, :] = image_z[100:, :, :]

        X4 = np.concatenate((X0, X1), axis=1)
        X5 = np.concatenate((X2, X3), axis=1)
        X6 = np.concatenate((X4, X5), axis=0)

        X6 = cv2.resize(X6, (256-32, 256))
        # print(X6.shape)

        X6 = np.concatenate((X6, X0a), axis=1)

        # print(X6.shape)
        # plt.figure()
        # plt.imshow(X6 / 255)
        # plt.show()

        num_file = base_name4 + 'grasping_result.csv'

        # print('num_file', num_file)
        # input('wait')

        with open(num_file, 'r') as f:
            df = pd.read_csv(f, header=None)
            # print(df.head)
            num_ball = df.iloc[0:1, 3:4].values
            num_ball1 = int(num_ball)
            # print('num_ball1', num_ball1)
            # input('wait')

            # After getting the combined image
            if num_ball1 == 0:
                count0 += 1
                data_0.append(num_ball1)
                image_0.append(X6)
            elif num_ball1 == 1:
                count1 += 1
                data_1.append(num_ball1)
                image_1.append(X6)
            elif num_ball1 == 2:
                count2 += 1
                data_2.append(num_ball1)
                image_2.append(X6)
            elif num_ball1 == 3:
                count3 += 1
                data_3.append(num_ball1)
                image_3.append(X6)
            elif num_ball1 == 4:
                count4 += 1
                data_4.append(num_ball1)
                image_4.append(X6)
            elif num_ball1 == 5:
                count5 += 1
                data_5.append(num_ball1)
                image_5.append(X6)
            elif num_ball1 == 6:
                count6 += 1
                data_6.append(num_ball1)
                image_6.append(X6)
            elif num_ball1 == 7:
                count7 += 1
                data_7.append(num_ball1)
                image_7.append(X6)

    print(
        '################################################ lengths #################################################')
    print(len(data_0), len(data_1), len(data_2), len(data_3), len(data_4), len(data_5), len(data_6), len(data_7))
    print(len(image_0), len(image_1), len(image_2), len(image_3), len(image_4), len(image_5), len(image_6),
          len(image_7))
    print(
        '################################################ lengths #################################################')

    # input('wait')

    ####################################################################################################################
    # Splitting
    ####################################################################################################################

    image_0_train = image_0[0:int(0.9 * 0.9 * len(image_0))]
    image_1_train = image_1[0:int(0.9 * 0.9 * len(image_1))]
    image_2_train = image_2[0:int(0.9 * 0.9 * len(image_2))]
    image_3_train = image_3[0:int(0.9 * 0.9 * len(image_3))]
    image_4_train = image_4[0:int(0.9 * 0.9 * len(image_4))]
    image_5_train = image_5[0:int(0.9 * 0.9 * len(image_5))]
    image_6_train = image_6[0:int(0.9 * 0.9 * len(image_6))]
    image_7_train = image_7[0:int(0.9 * 0.9 * len(image_7))]

    image_0_val = image_0[int(0.9 * 0.9 * len(image_0)):int(0.9 * len(image_0))]
    image_1_val = image_1[int(0.9 * 0.9 * len(image_1)):int(0.9 * len(image_1))]
    image_2_val = image_2[int(0.9 * 0.9 * len(image_2)):int(0.9 * len(image_2))]
    image_3_val = image_3[int(0.9 * 0.9 * len(image_3)):int(0.9 * len(image_3))]
    image_4_val = image_4[int(0.9 * 0.9 * len(image_4)):int(0.9 * len(image_4))]
    image_5_val = image_5[int(0.9 * 0.9 * len(image_5)):int(0.9 * len(image_5))]
    image_6_val = image_6[int(0.9 * 0.9 * len(image_6)):int(0.9 * len(image_6))]
    image_7_val = image_7[int(0.9 * 0.9 * len(image_7)):int(0.9 * len(image_7))]

    image_0_test = image_0[int(0.9 * len(image_0)):]
    image_1_test = image_1[int(0.9 * len(image_1)):]
    image_2_test = image_2[int(0.9 * len(image_2)):]
    image_3_test = image_3[int(0.9 * len(image_3)):]
    image_4_test = image_4[int(0.9 * len(image_4)):]
    image_5_test = image_5[int(0.9 * len(image_5)):]
    image_6_test = image_6[int(0.9 * len(image_6)):]
    image_7_test = image_7[int(0.9 * len(image_7)):]

    data_0_train = data_0[0:int(0.9 * 0.9 * len(data_0))]
    data_1_train = data_1[0:int(0.9 * 0.9 * len(data_1))]
    data_2_train = data_2[0:int(0.9 * 0.9 * len(data_2))]
    data_3_train = data_3[0:int(0.9 * 0.9 * len(data_3))]
    data_4_train = data_4[0:int(0.9 * 0.9 * len(data_4))]
    data_5_train = data_5[0:int(0.9 * 0.9 * len(data_5))]
    data_6_train = data_6[0:int(0.9 * 0.9 * len(data_6))]
    data_7_train = data_7[0:int(0.9 * 0.9 * len(data_7))]

    data_0_val = data_0[int(0.9 * 0.9 * len(data_0)):int(0.9 * len(data_0))]
    data_1_val = data_1[int(0.9 * 0.9 * len(data_1)):int(0.9 * len(data_1))]
    data_2_val = data_2[int(0.9 * 0.9 * len(data_2)):int(0.9 * len(data_2))]
    data_3_val = data_3[int(0.9 * 0.9 * len(data_3)):int(0.9 * len(data_3))]
    data_4_val = data_4[int(0.9 * 0.9 * len(data_4)):int(0.9 * len(data_4))]
    data_5_val = data_5[int(0.9 * 0.9 * len(data_5)):int(0.9 * len(data_5))]
    data_6_val = data_6[int(0.9 * 0.9 * len(data_6)):int(0.9 * len(data_6))]
    data_7_val = data_7[int(0.9 * 0.9 * len(data_7)):int(0.9 * len(data_7))]

    data_0_test = data_0[int(0.9 * len(data_0)):]
    data_1_test = data_1[int(0.9 * len(data_1)):]
    data_2_test = data_2[int(0.9 * len(data_2)):]
    data_3_test = data_3[int(0.9 * len(data_3)):]
    data_4_test = data_4[int(0.9 * len(data_4)):]
    data_5_test = data_5[int(0.9 * len(data_5)):]
    data_6_test = data_6[int(0.9 * len(data_6)):]
    data_7_test = data_7[int(0.9 * len(data_7)):]

    ####################################################################################################################
    # Balancing
    ####################################################################################################################

    total_num_train = max(len(data_0_train), len(data_1_train), len(data_2_train), len(data_3_train),
                          len(data_4_train), len(data_5_train), len(data_6_train), len(data_7_train))
    total_num_val = max(len(data_0_val), len(data_1_val), len(data_2_val), len(data_3_val), len(data_4_val),
                        len(data_5_val), len(data_6_val), len(data_7_val))
    total_num_test = max(len(data_0_test), len(data_1_test), len(data_2_test), len(data_3_test), len(data_4_test),
                         len(data_5_test), len(data_6_test), len(data_7_test))
    print('total_num_train, total_num_val, total_num_test', total_num_train, total_num_val, total_num_test)

    ################################################ train ###########################################################
    data_tmp = data_0_train.copy()
    image_tmp = image_0_train.copy()
    diff_0 = total_num_train - len(data_tmp)
    for k in range(diff_0):
        idx = k % len(data_tmp)
        data_0_train.append(data_tmp[idx])
        image_0_train.append(image_tmp[idx])

    data_tmp = data_1_train.copy()
    image_tmp = image_1_train.copy()
    diff_0 = total_num_train - len(data_tmp)
    if len(data_tmp) > 0:
        for k in range(diff_0):
            idx = k % len(data_tmp)
            # print('idx', idx)
            data_1_train.append(data_tmp[idx])
            image_1_train.append(image_tmp[idx])
            # print('len(data_1)', len(data_1))

    data_tmp = data_2_train.copy()
    image_tmp = image_2_train.copy()
    diff_0 = total_num_train - len(data_tmp)
    if len(data_tmp) > 0:
        for k in range(diff_0):
            idx = k % len(data_tmp)
            data_2_train.append(data_tmp[idx])
            image_2_train.append(image_tmp[idx])

    data_tmp = data_3_train.copy()
    image_tmp = image_3_train.copy()
    diff_0 = total_num_train - len(data_tmp)
    if len(data_tmp) > 0:
        for k in range(diff_0):
            idx = k % len(data_tmp)
            data_3_train.append(data_tmp[idx])
            image_3_train.append(image_tmp[idx])

    data_tmp = data_4_train.copy()
    image_tmp = image_4_train.copy()
    diff_0 = total_num_train - len(data_tmp)
    if len(data_tmp) > 0:
        for k in range(diff_0):
            idx = k % len(data_tmp)
            data_4_train.append(data_tmp[idx])
            image_4_train.append(image_tmp[idx])

    data_tmp = data_5_train.copy()
    image_tmp = image_5_train.copy()
    diff_0 = total_num_train - len(data_tmp)
    if len(data_tmp) > 0:
        for k in range(diff_0):
            idx = k % len(data_tmp)
            data_5_train.append(data_tmp[idx])
            image_5_train.append(image_tmp[idx])

    data_tmp = data_6_train.copy()
    image_tmp = image_6_train.copy()
    diff_0 = total_num_train - len(data_tmp)
    if len(data_tmp) > 0:
        for k in range(diff_0):
            idx = k % len(data_tmp)
            data_6_train.append(data_tmp[idx])
            image_6_train.append(image_tmp[idx])

    data_tmp = data_7_train.copy()
    image_tmp = image_7_train.copy()
    diff_0 = total_num_train - len(data_tmp)
    if len(data_tmp) > 0:
        for k in range(diff_0):
            idx = k % len(data_tmp)
            data_7_train.append(data_tmp[idx])
            image_7_train.append(image_tmp[idx])

    ################################################ val ###########################################################
    data_tmp = data_0_val.copy()
    image_tmp = image_0_val.copy()
    diff_0 = total_num_val - len(data_tmp)
    for k in range(diff_0):
        idx = k % len(data_tmp)
        data_0_val.append(data_tmp[idx])
        image_0_val.append(image_tmp[idx])

    data_tmp = data_1_val.copy()
    image_tmp = image_1_val.copy()
    diff_0 = total_num_val - len(data_tmp)
    if len(data_tmp) > 0:
        for k in range(diff_0):
            idx = k % len(data_tmp)
            # print('idx', idx)
            data_1_val.append(data_tmp[idx])
            image_1_val.append(image_tmp[idx])
            # print('len(data_1)', len(data_1))

    data_tmp = data_2_val.copy()
    image_tmp = image_2_val.copy()
    diff_0 = total_num_val - len(data_tmp)
    if len(data_tmp) > 0:
        for k in range(diff_0):
            idx = k % len(data_tmp)
            data_2_val.append(data_tmp[idx])
            image_2_val.append(image_tmp[idx])

    data_tmp = data_3_val.copy()
    image_tmp = image_3_val.copy()
    diff_0 = total_num_val - len(data_tmp)
    if len(data_tmp) > 0:
        for k in range(diff_0):
            idx = k % len(data_tmp)
            data_3_val.append(data_tmp[idx])
            image_3_val.append(image_tmp[idx])

    data_tmp = data_4_val.copy()
    image_tmp = image_4_val.copy()
    diff_0 = total_num_val - len(data_tmp)
    if len(data_tmp) > 0:
        for k in range(diff_0):
            idx = k % len(data_tmp)
            data_4_val.append(data_tmp[idx])
            image_4_val.append(image_tmp[idx])

    data_tmp = data_5_val.copy()
    image_tmp = image_5_val.copy()
    diff_0 = total_num_val - len(data_tmp)
    if len(data_tmp) > 0:
        for k in range(diff_0):
            idx = k % len(data_tmp)
            data_5_val.append(data_tmp[idx])
            image_5_val.append(image_tmp[idx])

    data_tmp = data_6_val.copy()
    image_tmp = image_6_val.copy()
    diff_0 = total_num_val - len(data_tmp)
    if len(data_tmp) > 0:
        for k in range(diff_0):
            idx = k % len(data_tmp)
            data_6_val.append(data_tmp[idx])
            image_6_val.append(image_tmp[idx])

    data_tmp = data_7_val.copy()
    image_tmp = image_7_val.copy()
    diff_0 = total_num_val - len(data_tmp)
    if len(data_tmp) > 0:
        for k in range(diff_0):
            idx = k % len(data_tmp)
            data_7_val.append(data_tmp[idx])
            image_7_val.append(image_tmp[idx])

    ################################################ test ###########################################################
    data_tmp = data_0_test.copy()
    image_tmp = image_0_test.copy()
    diff_0 = total_num_test - len(data_tmp)
    for k in range(diff_0):
        idx = k % len(data_tmp)
        data_0_test.append(data_tmp[idx])
        image_0_test.append(image_tmp[idx])

    data_tmp = data_1_test.copy()
    image_tmp = image_1_test.copy()
    diff_0 = total_num_test - len(data_tmp)
    if len(data_tmp) > 0:
        for k in range(diff_0):
            idx = k % len(data_tmp)
            # print('idx', idx)
            data_1_test.append(data_tmp[idx])
            image_1_test.append(image_tmp[idx])
            # print('len(data_1)', len(data_1))

    data_tmp = data_2_test.copy()
    image_tmp = image_2_test.copy()
    diff_0 = total_num_test - len(data_tmp)
    if len(data_tmp) > 0:
        for k in range(diff_0):
            idx = k % len(data_tmp)
            data_2_test.append(data_tmp[idx])
            image_2_test.append(image_tmp[idx])

    data_tmp = data_3_test.copy()
    image_tmp = image_3_test.copy()
    diff_0 = total_num_test - len(data_tmp)
    if len(data_tmp) > 0:
        for k in range(diff_0):
            idx = k % len(data_tmp)
            data_3_test.append(data_tmp[idx])
            image_3_test.append(image_tmp[idx])

    data_tmp = data_4_test.copy()
    image_tmp = image_4_test.copy()
    diff_0 = total_num_test - len(data_tmp)
    if len(data_tmp) > 0:
        for k in range(diff_0):
            idx = k % len(data_tmp)
            data_4_test.append(data_tmp[idx])
            image_4_test.append(image_tmp[idx])

    data_tmp = data_5_test.copy()
    image_tmp = image_5_test.copy()
    diff_0 = total_num_test - len(data_tmp)
    if len(data_tmp) > 0:
        for k in range(diff_0):
            idx = k % len(data_tmp)
            data_5_test.append(data_tmp[idx])
            image_5_test.append(image_tmp[idx])

    data_tmp = data_6_test.copy()
    image_tmp = image_6_test.copy()
    diff_0 = total_num_test - len(data_tmp)
    if len(data_tmp) > 0:
        for k in range(diff_0):
            idx = k % len(data_tmp)
            data_6_test.append(data_tmp[idx])
            image_6_test.append(image_tmp[idx])

    data_tmp = data_7_test.copy()
    image_tmp = image_7_test.copy()
    diff_0 = total_num_test - len(data_tmp)
    if len(data_tmp) > 0:
        for k in range(diff_0):
            idx = k % len(data_tmp)
            data_7_test.append(data_tmp[idx])
            image_7_test.append(image_tmp[idx])

    print(
        '################################################ lengths #################################################')
    print(len(data_0_train), len(data_1_train), len(data_2_train), len(data_3_train), len(data_4_train),
          len(data_5_train), len(data_6_train), len(data_7_train))
    print(len(image_0_train), len(image_1_train), len(image_2_train), len(image_3_train), len(image_4_train),
          len(image_5_train), len(image_6_train), len(image_7_train))
    print(len(data_0_val), len(data_1_val), len(data_2_val), len(data_3_val), len(data_4_val),
          len(data_5_val), len(data_6_val), len(data_7_val))
    print(len(image_0_val), len(image_1_val), len(image_2_val), len(image_3_val), len(image_4_val),
          len(image_5_val), len(image_6_val), len(image_7_val))
    print(len(data_0_test), len(data_1_test), len(data_2_test), len(data_3_test), len(data_4_test),
          len(data_5_test), len(data_6_test), len(data_7_test))
    print(len(image_0_test), len(image_1_test), len(image_2_test), len(image_3_test), len(image_4_test),
          len(image_5_test), len(image_6_test), len(image_7_test))
    print(
        '################################################ lengths #################################################')

    ####################################################################################################################
    # Combine
    ####################################################################################################################

    image_train = image_0_train + image_1_train + image_2_train + image_3_train + image_4_train + image_5_train + image_6_train + image_7_train
    image_val = image_0_val + image_1_val + image_2_val + image_3_val + image_4_val + image_5_val + image_6_val + image_7_val
    image_test = image_0_test + image_1_test + image_2_test + image_3_test + image_4_test + image_5_test + image_6_test + image_7_test

    data_train = data_0_train + data_1_train + data_2_train + data_3_train + data_4_train + \
                data_5_train + data_6_train + data_7_train
    data_val = data_0_val + data_1_val + data_2_val + data_3_val + data_4_val + \
                 data_5_val + data_6_val + data_7_val
    data_test = data_0_test + data_1_test + data_2_test + data_3_test + data_4_test + \
                data_5_test + data_6_test + data_7_test

    print(
        '******************************************************************************************************************')
    print("len(image_train)+len(image_val)+len(image_test)", len(image_train) + len(image_val) + len(image_test))
    print(
        '******************************************************************************************************************')

    print('len(image_train), len(data_train)', len(image_train), len(data_train))
    print('len(image_val), len(data_val)', len(image_val), len(data_val))
    print('len(image_test), len(data_test)', len(image_test), len(data_test))
    print('len(image_train) + len(image_val) + len(image_test)', len(image_train) + len(image_val) + len(image_test))

    print('np.max(data_train)', np.max(data_train), np.min(data_train))
    print('np.max(data_val)', np.max(data_val), np.min(data_val))
    print('np.max(data_test)', np.max(data_test), np.min(data_test))

    image_arr_train = np.array(image_train)
    num_arr_train = np.array(data_train)
    image_arr_val = np.array(image_val)
    num_arr_val = np.array(data_val)
    image_arr_test = np.array(image_test)
    num_arr_test = np.array(data_test)

    np.save(path_save + "image_arr_train", image_arr_train)
    np.save(path_save + "num_arr_train", num_arr_train)
    np.save(path_save + "image_arr_val", image_arr_val)
    np.save(path_save + "num_arr_val", num_arr_val)
    np.save(path_save + "image_arr_test", image_arr_test)
    np.save(path_save + "num_arr_test", num_arr_test)



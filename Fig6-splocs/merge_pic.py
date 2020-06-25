#!/usr/bin/env python
import numpy as np
from vtkplotter import *
import cv2

vp = Plotter(axes=0, bg='white', offscreen=True)  # shape="1/2"
settings.screenshotTransparentBackground = True
weight = 3840
height = 2880

# img =cv2.imread(file_path[i])
# img=cv2.hconcat([img,img,img])#水平拼接
# img=cv2.vconcat([img,img,img])


# merge picture
if 1:
    path1 = 'D:/project/ADMMAA/aa-lqlogisticre/code/code/mergedim'
    savepath = 'D:/project/ADMMAA/aa-lqlogisticre/code/code/merge/'
    sfid = 3
    for fid in range(sfid, sfid+3):
        for t in range(0, 1):
            name = '50/106_' + str(fid) + 'rAA6_DR6_t' + str(1) + '_outer_13.png'
            img1 = cv2.imread(path1 + name) #.x(0).y(height*t + (fid-1)*2*(height+10))
            print(np.shape(img1))
            print(path1 + name)

            name = '50/106_' + str(fid) + 'rAA6_DR6_t' + str(0) + '_outer_13.png'
            img2 = cv2.imread(path1+name) #.x(weight).y(height*t + (fid-1)*2*(height+10))
            print(np.shape(img2))
            print(path1 + name)

            name = '100/106_' + str(fid) + 'rAA6_DR6_t' + str(1) + '_outer_13.png'
            img3 = cv2.imread(path1 + name) #.x(weight*2).y(height*t + (fid-1)*2*(height+10))
            print(np.shape(img3))
            print(path1 + name)

            name = '100/106_' + str(fid) + 'rAA6_DR6_t' + str(0) + '_outer_13.png'
            img4 = cv2.imread(path1 + name) #.x(weight*2).y(height*t + (fid-1)*2*(height+10))
            tarimg = cv2.hconcat([img1, img2, img3, img4])
            # print(np.shape(img4))
            # print(path1 + name)

            if t==0 and fid == sfid:
                total_image = tarimg
            else:
                total_image = cv2.vconcat([total_image, tarimg])


    fout = savepath + str(sfid) + '_rmerge.png'
    cv2.imwrite(fout, total_image)

# merge picture with different stage
if 0:
    path1 = 'D:/project/ADMMAA/data/diffstage/fig'
    fid = 'monkey'
    outer = 1
    souter = 0
    for outer in (0, 1, 100, 101, 194, 195):
        for t in range(1, 2):
            name = '/f' + str(fid) + '_AA2_DR6_t' + str(t) + '_outer_' + str(outer) + '.png'
            img1 = cv2.imread(path1 + name) #.x(0).y(height*t + (fid-1)*2*(height+10))
            print(name)

            name = '/f' + str(fid) + '_mid_1' + str(t) + '_outer_' + str(outer) + '.png'
            img2 = cv2.imread(path1+name) #.x(weight).y(height*t + (fid-1)*2*(height+10))
            print(name)

            name = '/f' + str(fid) + '_mid_2' + str(t) + '_outer_' + str(outer) + '.png'
            img3 = cv2.imread(path1 + name) #.x(weight*2).y(height*t + (fid-1)*2*(height+10))
            tarimg = cv2.hconcat([img1, img2, img3])
            print(name)

            if t==1 and outer == souter:
                total_image = tarimg
            else:
                total_image = cv2.vconcat([total_image, tarimg])

            # name = 'file' + str(fid) + '/AA6_DR6_t' + str(t) + '_outer_0.png'
            # img = vp.load(path1 + name).x(0).y(height * t + (fid - 1) * 2 * (height + 10))
            # name = 'file' + str(fid) + '/mid_1' + str(t) + '_outer_0.png'
            # data = vp.load(path1 + name).x(weight).y(height * t + (fid - 1) * 2 * (height + 10))
            # name = 'file' + str(fid) + '/mid_2' + str(t) + '_outer_0.png'
            # data = vp.load(path1 + name).x(weight * 2).y(height * t + (fid - 1) * 2 * (height + 10))

    # vp.show(...)
    # vp.show(..., interactive=0)
    fout = path1 + str(fid) +  '_emerge2.png'
    cv2.imwrite(fout, total_image)
    # screenshot(fout, scale=5)
    # vp.clear()

## temp merge
if 0:
    path1 = 'D:/project/ADMMAA/sparseicp/merge/p08allf/'
    path2 = 'D:/project/ADMMAA/sparseicp/merge/p08dy/'
    path3 = 'D:/project/ADMMAA/sparseicp/merge/p08/'
    for fid in range(1, 10):
        name = path1 + 'imerge' + str(fid) + '.png'
        img1 = cv2.imread(name)
        print(np.shape(img1))
        print(name)

        name = path2 + 'imerge_f' + str(fid) + 'mu10.png'
        img2 = cv2.imread(name)
        print(np.shape(img2))
        print(name)

        tarimg = cv2.vconcat([img1, img2])

        fout = path3 + 'imerge_f' + str(fid) +'.png'
        cv2.imwrite(fout, tarimg)



# merge test mu
if 0:
    path1 = 'D:/project/ADMMAA/sparseicp/figp08nreset/1_f'
    path = 'D:/project/ADMMAA/sparseicp/merge/p08n/reset'
    sfid = 2

    for fid in range(sfid, sfid+5):
        for outer in range(10, 11):
            for t in range(1, 2):
                # name = str(fid) + 'rAA1_DR6_t' + str(1) + '_outer_1.png'
                # print(name)
                # img1 = cv2.imread(path1 + name) #.x(0).y(height*t + (fid-1)*(height+10))
                #
                # name = str(fid) + 'rAA1_DR6_t' + str(0) + '_outer_1.png'
                # print(name)
                # img2 = cv2.imread(path1+name) #.x(weight).y(height*t + (fid-1)*(height+10))
                #
                # name = str(fid) + 'eAA1_DR6_t' + str(1) + '_outer_1.png'
                # print(name)
                # img3 = cv2.imread(path1 + name) #.x(weight*2).y(height*t + (fid-1)*2*(height+10))
                #
                # name = str(fid) + 'eAA1_DR6_t' + str(0) + '_outer_1.png'
                # img4 = cv2.imread(path1 + name)  # .x(weight*2).y(height*t + (fid-1)*(height+10))
                # # tarimg = cv2.hconcat([img1, img2, img3])
                # tarimg = cv2.hconcat([img1, img2, img3, img4])

                name = str(fid) + '_mid_1' + str(t) + '_outer_' + str(outer) + '.png'
                print(name)
                img1 = cv2.imread(path1 + name)  # .x(0).y(height*t + (fid-1)*(height+10))
                print(np.size(img1))

                name = str(fid) + '_mid_2' + str(t) + '_outer_' + str(outer) + '.png'
                print(name)
                img2 = cv2.imread(path1 + name)  # .x(weight).y(height*t + (fid-1)*(height+10))
                print(np.size(img2))

                name = str(fid) + '_rAA6_DR6_t' + str(t) + '_outer_' + str(outer) + '_5.png'
                print(name)
                img3 = cv2.imread(path1 + name)  # .x(weight*2).y(height*t + (fid-1)*2*(height+10))
                print(np.size(img3))

                name = str(fid) + '_rAA6_DR6_t' + str(0) + '_outer_' + str(outer) + '_5.png'
                print(name)
                img4 = cv2.imread(path1 + name)
                print(np.size(img4))

                name = str(fid) + '_eAA6_DR6_t' + str(1) + '_outer_' + str(outer) + '_5.png'
                print(name)
                img5 = cv2.imread(path1 + name)
                print(np.size(img4))

                tarimg = cv2.hconcat([img1, img2, img3, img4, img5])

                if t == 1 and fid == sfid:
                    total_image = tarimg
                else:
                    total_image = cv2.vconcat([total_image, tarimg])

            # fout = path + 'merge_f' + str(fid) + 'mu' + str(outer) + '.png'
            # cv2.imwrite(fout, total_image)

        fout = path + '_emerge' + str(sfid) + '.png'
        cv2.imwrite(fout, total_image)
    # screenshot(fout, scale=5)
    # vp.clear()






from __future__ import print_function
import argparse
import numpy as np

import cv2
import time, glob, os



parser = argparse.ArgumentParser(description='ROI')
parser.add_argument('--filePath', type=str, default='./roi/roi_in/', help="path")
parser.add_argument('--name', type=str, default='butterfly.png', help='name')
parser.add_argument('--outPath', type=str, default='./roi/roi_out/', help='path')
parser.add_argument('--roi', nargs='+', type=int)
opt = parser.parse_args()
x,y,w,h=opt.roi
#使用鼠标获取ROI区域
image_list = glob.glob(opt.filePath + "/*.*")
length = len(image_list)
print(*image_list, sep='\n')
for i in range(length):
    t3=time.time()
    src = cv2.imread(image_list[i])
    cv2.rectangle(img=src,pt1=(x,y),pt2=(x+w,y+h),color=(0,0,255),thickness=2)

    img=src[y:y+h,x:x+w]

#cv2.imshow('img',img)
    _, name = os.path.split(image_list[i])
    cv2.imwrite(opt.outPath + name,img)
    cv2.imwrite(opt.outPath + 'ori_' + name ,src)
    
    t4=time.time()
    print("===> Processing: %s || Timer: %.4f sec." % (name, (t4 - t3)))
#src=cv2.imread(opt.filePath + opt.name)

#cv2.namedWindow('roi',cv2.WINDOW_NORMAL)

#cv2.imshow('roi',src)

#roi=cv2.selectROI(windowName="roi",img=src,showCrosshair=True,fromCenter=False)




#cv2.waitKey(0)

#cv2.destroyAllWindows()
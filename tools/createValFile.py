import os
import numpy as np

def inference(test_dir, inference_save_path):
    test_imgname_list = [os.path.join(test_dir, img_name) for img_name in os.listdir(test_dir)
                         if img_name.endswith(('.jpg', '.png', '.JPG','.jpeg', '.tif', '.tiff'))]
    assert len(test_imgname_list) != 0, 'test_dir has no imgs there.' \
                                        ' Note that, we only support img format of (.jpg, .png, and .tiff) '

    dropDet(test_imgname_list)

def dropDet(test_imgname_list):
    fnail_good = open('det_nail_good.txt', 'w')
    fnail_bad = open('det_nail_bad.txt', 'w')
    test_imgname_list.sort()
    for name in test_imgname_list:
        pathname = os.path.splitext(name)[0]  # 文件名
        fileName = pathname.split('/')[1]
        detfile = 'cropResult/' + str(fileName) + '/DropResult_nail.txt'
        with open(detfile, 'r') as f:
            lines = f.readlines()
        splitlines = [x.strip().split(' ') for x in lines]
        scores = np.array([float(x[4]) for x in splitlines])
        boxes = np.array([[int(z) for z in x[0:4]] for x in splitlines])
        labels=np.array([int(float(x[5])) for x in splitlines])
        for i in range(len(labels)):
            xmin, ymin, xmax, ymax = boxes[i]
            s = str(fileName) + ' '
            s=s+str(scores[i])+' '+str(xmin)+' '+str(ymin)+' '+str(xmax)+' '+str(ymax)+'\n'
            if labels[i]==1:
                fnail_good.write(s)
            if labels[i]==2:
                fnail_bad.write(s)
        print()


    fnail_good.close()
    fnail_bad.close()

if __name__ == '__main__':
    data_dir = 'testImage/'
    save_dir = 'inference_results/'
    inference(data_dir, save_dir)
import os
import numpy as np

def tst():#输出image.txt
    path = '/home/caoyu/Dataset/OTB100/'
    video_list = os.listdir(path)
    video_list.sort()
    for video in video_list:
        video_path = os.path.join(path, "{}/img/".format(video))   #100个序列的地址
        image_arry = os.listdir(video_path)
        image_arry.sort()
        save_path = os.path.join("/home/caoyu/1/{}.txt".format(video))

        f = open(save_path, 'w')
        for dir in image_arry:   # 把一个序列中的每个图片的jpg序号写进去
            img = video_path + dir
            f.write(img + '\n')

    f.close()

def tst2():#输出region.txt
    video_path = '/home/caoyu/Dataset/OTB100/'
    video_list = os.listdir(video_path)  #列出这个路径下的所有文件的名字
    video_list.sort()
    for video in video_list:
        gt_path = os.path.join(video_path, "{}/groundtruth_rect.txt".format(video))
        if video == 'Human4-2':
            gt = np.loadtxt(video_path + "{}/groundtruth_rect.2.txt".format(video), delimiter=(","), dtype=np.int)
        elif video in ['Vase','Twinnings','Sylvester','Surfer','Toy','Jogging', 'Jogging-1','Skating2-1','Skating2-2','Jogging-2','Rubik','Skating2', 'Singer1','Subway', 'Walking', 'Walking2', 'Woman']:
            gt = np.loadtxt(gt_path, delimiter=("\t"), dtype=np.int)
        else:
            gt = np.loadtxt(gt_path, delimiter=(","), dtype=np.int)
        x,y,w,h = gt[0]
        lt_x, lt_y, lb_x, lb_y, rt_x, rt_y, rb_x, r_y = x, y, x, y+h, x+w, y, x+w, y+h


        save_path = os.path.join("/home/caoyu/1/region/{}.txt".format(video))
        f = open(save_path, 'w')
        gt = "{},{},{},{},{},{},{},{}".format(lt_x, lt_y, lb_x, lb_y, rt_x, rt_y, rb_x, r_y)
        f.write(gt + '\n')
        f.close()


if __name__ == '__main__':
    tst2()

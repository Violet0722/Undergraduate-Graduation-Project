import cv2
import os
from matplotlib import pyplot as plt

def  save_bbox(image, bbox, gtbox, fram_num, seq_name, tracker_name, flag, s= None):

    img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    gt_x, gt_y, gt_w, gt_h = gtbox
    x, y, w, h = bbox
    path = "/home/day/1/{}/{}_{}/".format(seq_name, seq_name, tracker_name)
    #print(path + "{}{}".format(fram_num,".png"))
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)
    cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 2)
    cv2.rectangle(img, (int(gt_x), int(gt_y)), (int(gt_x + gt_w), int(gt_y + gt_h)), (0, 0, 255), 2)
    cv2.putText(img, flag, (int(bbox[0]), int(bbox[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    if tracker_name == "atom":
        pass
    else:
        sx, sy, sw, sh = s
        cv2.rectangle(img, (int(sx), int(sy)), (int(sx + sw), int(sy + sh)), (0, 255, 255), 2)
    cv2.imwrite(path + "{}{}".format(fram_num,".png"), img)


def save_heat_map(score_map, fram_num, seq_name, tracker_name):
    plt.imshow(score_map)
    fig = plt.gcf()
    plt.margins(0, 0)
    path = "/home/day/1/{}/{}_{}_heatmap/".format(seq_name, seq_name, tracker_name)
    # print(path + "{}{}".format(fram_num,".png"))
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)
    fig.savefig(path + "{}{}".format(fram_num,".png"), dpi=500, bbox_inches='tight')  # dpi越高越清晰


#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import os


def convert_bbox(bbox_infos):
	'''Convert x1y1wh to normalize x_center y_center w h'''
    x1, y1, w, h = int(bbox_infos[0]), int(bbox_infos[1]), int(bbox_infos[2]), int(bbox_infos[3])
    x_center = x1+w/2
    y_center = y1+h/2
    x_center /=1024
    y_center /=540
    w /= 1024
    h /= 540
    return [str(x_center), str(y_center), str(w), str(h)]

def get_info_from_line(infos):
	'''Extract raw info from a single line in annotation file'''
    x_center, y_center, w, h = convert_bbox([infos['<bbox_left>'], infos['<bbox_top>'], infos['<bbox_width>'], infos['<bbox_height>']])
    # 0- car,  1 - truck, 2- bus
    class_idx = str(infos['<object_category>']-1)
    return [x_center,y_center,w,h, class_idx]

def get_annot_data(txt_file):
	'''Read annotation into a Pandas dataframe'''
    annot_data =  pd.read_csv(txt_file, delimiter=',', names=['<frame_index>','<target_id>','<bbox_left>','<bbox_top>','<bbox_width>','<bbox_height>','<out-of-view>','<occlusion>','<object_category>'])
    return annot_data

def get_infos(frame_id):
	'''Extract sub-dataframe w.r.t frame_id - get only box coordiantes and object caterogy'''
    infos = annot_data[annot_data['<frame_index>'] == frame_id].loc[:,['<bbox_left>','<bbox_top>','<bbox_width>','<bbox_height>','<object_category>']]
    return infos

def create_annot(img_path, infos):
	'''Create annotation text file from each sub-data frame'''
    assert os.path.exists(img_path) 
    annot_path= img_path[:-3] + 'txt'
    print(annot_path)
    with open(annot_path, 'w') as f: 
        for row in infos.iterrows(): #Fix
            #print(get_info_from_line(row))
            f.write(' '.join(info for info in get_info_from_line(row[1])))
            f.write('\n')

if __name__=="__main__":

	root_dir = '../Darknet/UAV-benchmark-M' #Your path to dataset
	for txt_file in os.listdir('.'):
	    if 'txt' in txt_file:
	        with open(txt_file, 'r') as f:
	            annot_data =  get_annot_data(txt_file)
	            image_folder = txt_file[:5]
	            #with open('d:/data.txt', 'a') as f:
	            for image in os.listdir(os.path.join(root_dir, image_folder)):
	                    if '.jpg' in image:
	                        frame_id = int(image[3:-4])
	                        infos = get_infos(frame_id)
	                        img_path= f'{root_dir}/{image_folder}/{image}'
	                        create_annot(img_path, infos)




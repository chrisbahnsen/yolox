# https://github.com/roboflow-ai/YOLOX/blob/main/voc_txt.py
import os
import random
import sys
from pathlib import Path
import argparse
import tqdm

def shufflelines(filepath):
    lines = open(filepath, 'r').readlines()
    random.shuffle(lines)
    open(filepath, 'w').writelines(lines)

def convertvoc(root_path):

    xmlfilepath = root_path + 'VOC2007/Annotations/'
    os.makedirs(xmlfilepath, exist_ok = True)
    imagefilepath = root_path + 'VOC2007/JPEGImages/'
    os.makedirs(imagefilepath, exist_ok = True)

    filenames = os.listdir(root_path)

    # Move annotations to annotations folder
    for filename in tqdm(filenames, "Moving files to VOC2007 folder"):
        if filename.endswith('.xml'):
            with open(os.path.join(root_path, filename)) as f:
                Path(root_path + filename).rename(xmlfilepath + filename)

        if filename.endswith('.jpg'):
            with open(os.path.join(root_path, filename)) as f:
                Path(root_path + filename).rename(imagefilepath + filename)


    txtsavepath = root_path + '/VOC2007/ImageSets/Main'

    if not os.path.exists(root_path):
        print("cannot find such directory: " + root_path)
        exit()

    if not os.path.exists(txtsavepath):
        os.makedirs(txtsavepath)

    trainval_percent = 0.9
    train_percent = 0.8
    total_xml = os.listdir(xmlfilepath)

    # In order to make sure that we get a decent split across classes with varying
    # number of images per class, we make the train/test/val split for each class
    # In order to do so, we form a list of classes based on the xml file names
    # The base assumption there is that the xml file is named in the following manner:
    # class_name_prefix.xml -> then we will decode "class_name" as the class name.
    # This will not work with xml files named as the follows:
    # class_name_prefix_and_other_info.xml -> "class_name_prefix_and_other" -> class name

    xml_grouped_by_class = dict()

    for x in total_xml:
        spl = x.split('_')    
        class_name = '_'.join(spl[:-1])

        if not class_name in xml_grouped_by_class:
            xml_grouped_by_class[class_name] = [x]
        else:
            xml_grouped_by_class[class_name].append(x)

    ptrainval = txtsavepath + '/trainval.txt'
    ptest = txtsavepath + '/test.txt'
    ptrain = txtsavepath + '/train.txt'
    pval =  txtsavepath + '/val.txt'

    ftrainval = open(ptrainval, 'w')
    ftest = open(ptest, 'w')
    ftrain = open(ptrain, 'w')
    fval = open(pval, 'w')

    total_train_val = 0
    total_train = 0
    total_test = 0


    for cls_name, x_list in tqdm(xml_grouped_by_class.items(), "Creating train/test split"):
        num = len(x_list)
        list = range(num)
        tv = int(num * trainval_percent)
        tr = int(tv * train_percent)
        trainval = random.sample(list, tv)
        train = random.sample(trainval, tr)

        total_train_val += tv
        total_train += tr
        total_test += num - tv

        for i in list:
            name = x_list[i][:-4] + '\n'
            if i in trainval:
                ftrainval.write(name)
                if i in train:
                    ftrain.write(name)
                else:
                    fval.write(name)
            else:
                ftest.write(name)

    
    print("Train and val size:", total_train_val)
    print("Train size:", total_train)
    print("Test size:", total_test)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()

    # Re-open the files and shuffle the order of the files such that the files are not ordered by class
    shufflelines(ptrainval)
    shufflelines(ptest)
    shufflelines(ptrain)
    shufflelines(pval)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Make the final conversion to VOC dataset format')
    parser.add_argument('--sourcepath',type = str, default = 'dataset/', help ='Source path of the conversion')

    args = parser.parse_args()

    convertvoc(args.sourcepath)
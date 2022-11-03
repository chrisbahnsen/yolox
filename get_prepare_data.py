import csv
import os
import argparse
from oidv6 import OIDv6
from oidv6_to_voc import convertFromOidv6ToVoc
from voc_txt import convertvoc


def getPrepareData(model):

    classesDownloadList = []
    fullClassInfo = []

    classRenameDict = dict()
    renameFieldnames = ['class_name', 'renamed_class_name']

    # with open('renamed_class_names.csv', 'r') as f:
    #     reader = csv.DictReader(f)
    #     renameFieldnames = reader.fieldnames

    #     for row in reader:
    #         classRenameDict[row['class_name']] = row['renamed_class_name']

    # Open the class information file for the model 
    with open(model + '.csv', 'r') as f:
        reader = csv.DictReader(f)

        # Populate a list of all the classes that should be downloaded from 
        # Google OpenImages V6
        for row in reader:
            classesDownloadList.append(row['class_name'])

            if row['renamed_class_name']:
                classRenameDict[row['class_name']] = row['renamed_class_name']
            else:
                classRenameDict[row['class_name']] = row['class_name']

            fullClassInfo.append(row)

    with open(model + '.txt', 'w') as f:
        for c in classesDownloadList:
            f.write(c + '\n')


    with open('renamed_class_names.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=renameFieldnames)
        writer.writeheader()

        for c in classRenameDict.items():
            v = {'class_name': c[0], 'renamed_class_name': c[1]}
            writer.writerow(v)

    # # Now download the data
    # # ...move this to a shell script

    args = dict()
    args['type_data'] = 'all'
    args['classes'] = [model + '.txt']
    args['limit'] = 20
    args['multi_classes'] = True
    args['dataset'] = model
    args['yes'] = True
    args['no_labels'] = False
    args['no_clear_shell'] = True
    args['command'] = 'downloader'

    oid = OIDv6.OIDv6()
    #oid.download(args)

    # For the categories with amount of data below 3000, merge the test and validation set into the training set
    for i in fullClassInfo:
        try:
            if int(i['train']) < 3000:
                print('Combining train, test and validation imags for class: ' + i['class_name'])
                # Now move data from test to train
                
                # First images
                os.system('find ' + model + '/multidata/test/ -name \'' + i['class_name'] + '_*.jpg\' -exec mv \'{}\' ' + model + '/multidata/train/ \;')

                # Then labels
                os.system('find ' + model + '/multidata/test/labels -name \'' + i['class_name'] + '_*.txt\' -exec mv \'{}\' ' + model + '/multidata/train/labels/ \;')
                        #'find \'OIDv6/multidata/test/ -name ' + i['class_name'] + '_*.jpg\' -exec mv \'{}\' OIDv6/multidata/train/ \;' 

                # And from val to train
                os.system('find ' + model + '/multidata/validation/ -name \'' + i['class_name'] + '_*.jpg\' -exec mv \'{}\' ' + model + '/multidata/train/ \;')
                os.system('find ' + model + '/multidata/validation/labels -name \'' + i['class_name'] + '_*.txt\' -exec mv \'{}\' ' + model + '/multidata/train/labels/ \;')
        except Exception as e:
            print(e)

    # Combine the newly acquired data

    # Call the OIDV6 script from here
    convertFromOidv6ToVoc(model + '/multidata/train', 
                        model + '/multidata/train',
                        'renamed_class_names.csv')

    # Copy to separate VOC folder
    os.system('mkdir -p datasets/' + model + '/VOCdevkit/')
    os.system('find ' + model + '/multidata/train/ -type f -name \'*.jpg\' -print0 | xargs -0 mv -t datasets/' + model + '/VOCdevkit/')
    os.system('find ' + model + '/multidata/train/ -type f -name \'*.xml\' -print0 | xargs -0 mv -t datasets/' + model + '/VOCdevkit/')

    os.system('mkdir -p datasets/' + model + '/VOCdevkit/VOC2007/')

    # Make the final conversion to VOC
    convertvoc('datasets/' + model + '/VOCdevkit/')

    print("Copying files...")
    os.system('mkdir -p datasets/' + model + '/VOCdevkit/VOC2012')
    os.system('cp -r datasets/' + model + '/VOCdevkit/VOC2007/. datasets/' + model + '/VOCdevkit/VOC2012')


    # Create the class files in COCO and VOC format
    finalClassNames = set()

    for oldClassName, finalClassName in classRenameDict.items():
        finalClassNames.add(finalClassName)

    with open('yolox/data/datasets/voc_classes.py', 'w') as f:
        f.write('VOC_CLASSES = (\n')
        for className in finalClassNames:
            f.write('    \"' + className + '\",\n')
        
        f.write(')')

    with open('yolox/data/datasets/coco_classes.py', 'w') as f:
        f.write('COCO_CLASSES = (\n')
        for className in finalClassNames:
            f.write('    \"' + className + '\",\n')
        
        f.write(')')


    # Create the training script
    print("NUM_CLASSES: " + str(len(finalClassNames)))

    scriptFile = 'train-' + model + '.sh'

    with open(scriptFile, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('python3.9 tools/train.py -f exps/example/yolox_voc/yolox_voc_nano_custom.py -d 1 -b 8 --fp16 -c yolox_nano.pth -a ' + model + ' -u ' + str(len(finalClassNames)))

    os.system('chmod +x ' + scriptFile)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Utility script for downloading and preparing dataset for YOLOX training")
    parser.add_argument('--model', type=str, default="firstwords", help="Name of the model. Make sure that appropriate csv file is provided")

    args = parser.parse_args()

    print(args)

    getPrepareData(args.model)


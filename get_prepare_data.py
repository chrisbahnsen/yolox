import csv
import os
import argparse
from oidv6 import OIDv6
from oidv6_to_voc import convertFromOidv6ToVoc
from voc_txt import convertvoc
from pathlib import Path
from shutil import move
from os.path import basename, join
import tqdm

def getPrepareData(model, limit):

    classesDownloadList = []
    fullClassInfo = dict()

    classRenameDict = dict()
    renameFieldnames = ['class_name', 'renamed_class_name']

    # Open the class information file for the model 
    with open(model + '.csv', 'r', encoding='utf-8-sig') as f:
        sniffer = csv.Sniffer()

        dialect = sniffer.sniff(f.read(1024), delimiters=';,')
        f.seek(0)
        reader = csv.DictReader(f, dialect=dialect)

        # Populate a list of all the classes that should be downloaded from 
        # Google OpenImages V6
        for row in reader:
            download = True

            if 'database' in row:
                download = False

                if 'OID' in row['database']:
                    download = True
            
            if download:
                classesDownloadList.append(row['class_name'])

                if row['renamed_class_name']:
                    classRenameDict[row['class_name']] = row['renamed_class_name']
                else:
                    classRenameDict[row['class_name']] = row['class_name']

            fullClassInfo[row['class_name']] = row

    with open('renamed_class_names.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=renameFieldnames)
        writer.writeheader()

        for c in classRenameDict.items():
            v = {'class_name': c[0], 'renamed_class_name': c[1]}
            writer.writerow(v)

    # Search for images that are already downloaded to 
    # another directory
    os.makedirs('datasets/' + model + '/VOCdevkit/VOC2007/', exist_ok=True)

    for cls in classesDownloadList:
        clsName = cls.replace(' ', '_')
        matches = []

        for path in Path('datasets').rglob(clsName + "*"):
            #print(path)
            matches.append(str(path))

        # We find both jpg and xml files, which count as one
        numMatches = len(matches) / 2

        if numMatches >= int(fullClassInfo[cls]['train']):
            # No need to re-download this class, plenty of existing data
            # already
            classesDownloadList.remove(cls)
            print("Found {} images of class {}, no need to re-download".format(numMatches, cls))

            # Now move this data
            for m in tqdm(matches, 'Copying images to {}'.format(join('datasets', model, 'VOCdevkit'))):
                # We only move the 2007 data as we will copy the 2012 data later
                if ('2007' in m or '2012' in m) and model not in m:
                    d = join('datasets', model, 'VOCdevkit', basename(m))
                    move(m, d)

    with open(model + '.txt', 'w') as f:
        for c in classesDownloadList:
            f.write(c + '\n')


    # # Now download the data
    # # ...move this to a shell script

    args = dict()
    args['type_data'] = 'all'
    args['classes'] = [model + '.txt']
    args['limit'] = limit
    args['multi_classes'] = True
    args['dataset'] = model
    args['yes'] = True
    args['no_labels'] = False
    args['no_clear_shell'] = True
    args['command'] = 'downloader'

    oid = OIDv6.OIDv6()
    oid.download(args)

    # For the categories with amount of data below 3000, merge the test and validation set into the training set
    for className, i in fullClassInfo.items():
        try:
            if int(i['train']) < limit:
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

    # # Combine the newly acquired data

    # # Call the OIDV6 script from here
    convertFromOidv6ToVoc(model + '/multidata/train', 
                        model + '/multidata/train',
                        'renamed_class_names.csv')

    # Copy to separate VOC folder
    os.makedirs('datasets/' + model + '/VOCdevkit/', exist_ok=True)

    print("Moving data, hold tight...")
    if os.name == 'nt':
        cwd = os.getcwd()
        os.system('powershell Move-Item -Path ' + cwd + '\\' +  model + '\\multidata\\train\\*.jpg -Destination ' + cwd + '\\datasets\\' + model + '\\VOCdevkit\\')
        os.system('powershell Move-Item -Path ' + cwd + '\\' +  model + '\\multidata\\train\\*.xml -Destination ' + cwd + '\\datasets\\' + model + '\\VOCdevkit\\')
    else:
        os.system('find ' + model + '/multidata/train/ -type f -name \'*.jpg\' -print0 | xargs -0 mv -t datasets/' + model + '/VOCdevkit/')
        os.system('find ' + model + '/multidata/train/ -type f -name \'*.xml\' -print0 | xargs -0 mv -t datasets/' + model + '/VOCdevkit/')

    os.makedirs('datasets/' + model + '/VOCdevkit/VOC2007/', exist_ok=True)

    # Make the final conversion to VOC
    print("Making the final conversion to VOC, creating trainval and test samples")
    convertvoc('datasets/' + model + '/VOCdevkit/')

    print("Copying files...")
    os.makedirs('datasets/' + model + '/VOCdevkit/VOC2012/', exist_ok=True)
    
    if os.name == 'nt':
        os.system('powershell Copy-Item -Path ' + cwd + '\\datasets\\' +  model + '\\VOCdevkit\\VOC2007\\* -Destination ' + cwd + '\\datasets\\' + model + '\\VOCdevkit\\VOC2012\\ -Recurse')
    else:
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
    print("Number of classes: " + str(len(finalClassNames)))
    

    scriptFile = 'train-' + model + '.sh'

    with open(scriptFile, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('python3.9 tools/train.py -f exps/example/yolox_voc/yolox_voc_nano_custom.py -d 1 -b 8 --fp16 -c yolox_nano.pth -a ' + model + ' -u ' + str(len(finalClassNames)))

    os.system('chmod +x ' + scriptFile)

    print("Run {} to start training".format(scriptFile))

    ## TODO: Write script for evaluation here


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Utility script for downloading and preparing dataset for YOLOX training")
    parser.add_argument('--model', type=str, default="animal", help="Name of the model. Make sure that appropriate csv file is provided")
    parser.add_argument('--limit', type=int, default="3000", help="Maximum number of images per class that are downloaded")

    args = parser.parse_args()

    print(args)

    getPrepareData(args.model, args.limit)


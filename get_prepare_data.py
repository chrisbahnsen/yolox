import csv
import os
import argparse
from pathlib import Path
from shutil import move
from os.path import basename, join
from tqdm import tqdm
from xmltodict import unparse, parse
from oidv6 import OIDv6

from oidv6_to_voc import convertFromOidv6ToVoc
from voc_txt import convertvoc
from downloadlvisdata import getLVISbyCategories



def getPrepareData(model, limit):

    downloadList = {'oid': []}
    fullClassInfo = dict()

    classRenameDict = dict()
    csvPath = 'datasets/{}.csv'.format(model)

    # Open the class information file for the model 
    with open(csvPath, 'r', encoding='utf-8-sig') as f:
        sniffer = csv.Sniffer()

        dialect = sniffer.sniff(f.read(1024), delimiters=';,')
        f.seek(0)
        reader = csv.DictReader(f, dialect=dialect)

        # Populate a list of all the classes that should be downloaded from 
        # Google OpenImages V6, LVIS, or PASCAL-parts
        for row in reader:
            database = 'oid'

            if 'database' in row:
                database = row['database'].lower()

                if database not in downloadList:
                    downloadList[database] = []
            
            className = ""

            if 'class_name' in row:
                className = row['class_name']
            elif 'class' in row:
                className = row['class']
                row['class_name'] = row['class']
            else:
                raise RuntimeError("The file {} should contain a column named \'class\'".format(csvPath))

            # Remove leading or trailing spaces from the name
            className = className.strip()
            className = className.lower()

            downloadList[database].append(className)    

            renamedClassName = ""

            if 'renamed_class_name' in row:
                renamedClassName = row['renamed_class_name']
            elif 'Rename' in row:
                renamedClassName = row['Rename']
            elif 'rename' in row:
                renamedClassName = row['rename']
            else:
                raise RuntimeError("The file {} should contain a column named \'rename\'".format(csvPath))

            renamedClassName = renamedClassName.strip()
            renamedClassName = renamedClassName.lower()

            if renamedClassName:
                classRenameDict[className.replace('_', ' ')] = renamedClassName.replace('_', ' ')
            else:
                classRenameDict[className.replace('_', ' ')] = className.replace('_', ' ')

            fullClassInfo[className] = row

    # Search for images that are already downloaded to 
    # another directory
    os.makedirs('datasets/' + model + '/VOCdevkit/VOC2007/Annotations', exist_ok=True)
    os.makedirs('datasets/' + model + '/VOCdevkit/VOC2007/JPEGImages', exist_ok=True)

    prunedDownloadList = []

    for cls in downloadList['oid']:
        clsName = cls.replace(' ', '_')
        matches = []

        for path in Path('datasets').rglob(clsName + "*"):
            matches.append(str(path))

        # We find both jpg and xml files, which count as one
        numMatches = int(len(matches) / 2)

        # There might be errors in the numbers reported in the spreadsheet, 
        # so in order to avoid too many re-downloads, we relax this criteria
        trainImgs = int(float(fullClassInfo[cls]['train'])* 0.4)

        if numMatches >= trainImgs or numMatches >= limit:
            # No need to re-download this class, plenty of existing data
            # already
            print("Found {} images of class {}, no need to re-download".format(numMatches, cls))

            # Now move this data
            for m in tqdm(matches, 'Copying images to {}'.format(join('datasets', model, 'VOCdevkit'))):
                # We only move the 2007 data as we will copy the 2012 data later
                if ('2007' in m or '2012' in m) and model not in m:
                    if 'xml' in m:
                        d = join('datasets', model, 'VOCdevkit', 'VOC2007', 'Annotations', basename(m))
                    else: 
                        d = join('datasets', model, 'VOCdevkit', 'VOC2007', 'JPEGImages', basename(m))

                    move(m, d)
        else:
            prunedDownloadList.append(cls)


    classListPath = os.path.join('datasets', "{}.txt".format(model))

    with open(classListPath, 'w') as f:
        for c in prunedDownloadList:
            f.write(c + '\n')


    # # Now download the data
    args = dict()
    args['type_data'] = 'all'
    args['classes'] = [classListPath]
    args['limit'] = limit
    args['multi_classes'] = True
    args['dataset'] = model
    args['yes'] = True
    args['no_labels'] = False
    args['no_clear_shell'] = True
    args['command'] = 'downloader'

    oid = OIDv6.OIDv6()
    oid.download(args)

    # For the categories with amount of data below limit, merge the test and validation set into the training set
    for className, i in fullClassInfo.items():
        try:
            trainImgs = int(float(i['train']))

            if trainImgs < limit:
                print('Combining train, test and validation imags for class: ' + i['class_name'])
                # Now move data from test to train
                
                if os.name == 'nt':
                    os.system('powershell Move-Item -Path ' + cwd + '\\' +  model + '\\multidata\\test\\*.jpg -Destination ' + cwd + '\\' + model + '\\multidata\\train\\')
                    os.system('powershell Move-Item -Path ' + cwd + '\\' +  model + '\\multidata\\test\\labels\\*.txt -Destination ' + cwd + '\\' + model + '\\multidata\\train\\labels\\')

                    os.system('powershell Move-Item -Path ' + cwd + '\\' +  model + '\\multidata\\validation\\*.jpg -Destination ' + cwd + '\\' + model + '\\multidata\\train\\')
                    os.system('powershell Move-Item -Path ' + cwd + '\\' +  model + '\\multidata\\validation\\labels\\*.txt -Destination ' + cwd + '\\' + model + '\\multidata\\train\\labels\\')
                else:
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
    oidTrainPath = os.path.join(model + '/multidata/train')

    convertFromOidv6ToVoc(os.path.join(oidTrainPath, 'labels'), 
                        oidTrainPath,
                        oidTrainPath,
                        classRenameDict, 
                        deleteImagesWithNoAnn=False)

    # And make sure that files that was found elsewhere are properly renamed, too
    convertFromOidv6ToVoc(model + '/multidata/train/labels', 
                         'datasets/' + model + '/VOCdevkit/VOC2007/JPEGImages',
                         'datasets/' + model + '/VOCdevkit/VOC2007/Annotations',
                         classRenameDict,
                         deleteImagesWithNoAnn=False)


    # Copy to separate VOC folder
    print("Moving data, hold tight...")
    if os.name == 'nt':
        cwd = os.getcwd()
        os.system('powershell Move-Item -Path ' + cwd + '\\' +  model + '\\multidata\\train\\*.jpg -Destination ' + cwd + '\\datasets\\' + model + '\\VOCdevkit\\VOC2007\\JPEGImages\\')
        os.system('powershell Move-Item -Path ' + cwd + '\\' +  model + '\\multidata\\train\\*.xml -Destination ' + cwd + '\\datasets\\' + model + '\\VOCdevkit\\VOC2007\\Annotations\\')
    else:
        os.system('find ' + model + '/multidata/train/ -type f -name \'*.jpg\' -print0 | xargs -0 mv -t datasets/' + model + '/VOCdevkit/VOC2007/JPEGImages')
        os.system('find ' + model + '/multidata/train/ -type f -name \'*.xml\' -print0 | xargs -0 mv -t datasets/' + model + '/VOCdevkit/VOC2007/Annotations')

    voc2007dir = 'datasets/' + model + '/VOCdevkit/VOC2007/'
    # Download LVIS categories, if any
    if 'lvis' in downloadList:
        if len(downloadList['lvis']) > 0:
            print("---- Downloading LVIS data ------")
            getLVISbyCategories('val', downloadList['lvis'], voc2007dir, classRenameDict, limit)
            getLVISbyCategories('train', downloadList['lvis'], voc2007dir, classRenameDict, limit)

    finalClassNames = set()

    for oldClassName, finalClassName in classRenameDict.items():
        finalClassNames.add(finalClassName)


    # Check if there are annotations with classes that we don't support
    annPath = os.path.join(voc2007dir, 'Annotations')
    allAnns = os.listdir(annPath)

    for a in tqdm(allAnns, "Checking annotations"):

        deleteAnn = False
        renameAnn = False
        ann = []
        
        with open(os.path.join(annPath, a), 'r') as f:
            ann = parse(f.read())

            if type(ann['annotation']['object']) == dict:
                name = ann['annotation']['object']['name'] 
                if name not in finalClassNames:
                    # Try to check if we can rename 
                    # this object
                    if name in classRenameDict:
                        ann['annotation']['object']['name'] = classRenameDict[name]
                        renameAnn = True
                    else:
                        deleteAnn = True
            else:
                # List of dicts
                for object in ann['annotation']['object']:
                    if object['name'] not in finalClassNames:
                        # Try to check if we can rename 
                        # this object
                        if object['name'] in classRenameDict:
                            object['name'] = classRenameDict[object['name']]
                            renameAnn = True
                        else:
                            deleteAnn = True

        if deleteAnn:
            print("Deleting, not in class list: {}".format(a))
            try:
                os.remove(os.path.join(annPath, a))
                os.remove(os.path.join(voc2007dir, 'JPEGImages', a.replace('.xml', '.jpg')))
            except OSError:
                pass
                    
        if renameAnn:
            # Save the renamed annotation
            with open(os.path.join(annPath, a), 'w') as f:
                unparse(ann, f, full_document=False, pretty=True)         



    # Make the final conversion to VOC
    print("Making the final conversion to VOC, creating trainval and test samples")
    convertvoc('datasets/' + model + '/VOCdevkit/')


    # Create the class files in COCO and VOC format
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

    # Write the class list for consumption by NatML
    classListPath = os.path.join('datasets', "{}-NatML.txt".format(model))
    
    with open(classListPath, 'w') as f:
        for className in finalClassNames:
            f.write('{}\n'.format(className))


    # Create the training script
    print("Number of classes: " + str(len(finalClassNames)))
    specificNano = []
    specificNanoPath = 'exps/example/yolox_voc/{}.py'.format(model)

    # Copy the nano_custom file, change number of classes
    with open('exps/example/yolox_voc/yolox_voc_nano_custom.py') as f:
        nano = f.readlines()

        firstEncounter = True

        for line in nano:
            if firstEncounter and 'self.num_classes' in line:
                firstEncounter = False
                mod = '        self.num_classes = {}'.format(len(finalClassNames))
                specificNano.append(mod)
            else:
                specificNano.append(line)

    with open(specificNanoPath, 'w') as f:
        f.writelines(specificNano)

    if os.name == 'nt':
        scriptExt =  '.bat'
    else:
        scriptExt =  '.sh'


    scriptFile = 'train-' + model + scriptExt  

    with open(scriptFile, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('python tools/train.py -f exps/example/yolox_voc/yolox_voc_nano_custom.py -d 1 -b 16 --fp16 -c yolox_nano.pth -a ' 
                + model + ' -u ' + str(len(finalClassNames)) + ' --logger wandb wandb-project ' + model)

    os.system('chmod +x ' + scriptFile)

    print("Run {} to start training".format(scriptFile))

    scriptFile = 'evaluate-' + model + scriptExt

    with open(scriptFile, 'w') as f:
        epochPath = 'YOLOX-outputs/{}/latest_ckpt.pth'.format(model)

        f.write('#!/bin/bash\n')
        f.write('python tools/eval.py -n {} -c {} -b 4 -d 1 --conf 0.01 -f {}'.format(model, epochPath, specificNanoPath))

    print("Run {} to evaluate the trained model".format(scriptFile))

    scriptFile = 'convert-' + model + '-toTFLite' + scriptExt

    with open(scriptFile, 'w') as f:
        epochPath = 'YOLOX-outputs/{}/latest_ckpt.pth'.format(model)
        onnxPath = 'YOLOX-outputs/{}/{}.onnx'.format(model, model)

        f.write('#!/bin/bash\n')
        f.write('python tools/export_onnx.py --output_name {} -f {} -c {}'.format(onnxPath, specificNanoPath, epochPath))
        f.write('python tools/convertToTflite.py --modelPath {onnxPath}')



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Utility script for downloading and preparing dataset for YOLOX training")
    parser.add_argument('--model', type=str, default="inde", help="Name of the model. Make sure that appropriate csv file is provided")
    parser.add_argument('--limit', type=int, default="5000", help="Maximum number of images per class that are downloaded")

    args = parser.parse_args()

    getPrepareData(args.model, args.limit)


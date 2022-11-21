import lvis
import zipfile
import os
from tqdm import tqdm
from xmltodict import unparse
import wget
import time

BBOX_OFFSET = 0


def base_dict(filename, width, height, depth=3):
    return {
        "annotation": {
            "filename": os.path.split(filename)[-1],
            "folder": "VOCCOCO", "segmented": "0", "owner": {"name": "unknown"},
            "source": {'database': "The COCO 2017 database", 'annotation': "COCO 2017", "image": "unknown"},
            "size": {'width': width, 'height': height, "depth": depth},
            "object": []
        }
    }


def base_object(size_info, name, bbox):
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h

    width = size_info['width']
    height = size_info['height']

    x1 = max(x1, 0) + BBOX_OFFSET
    y1 = max(y1, 0) + BBOX_OFFSET
    x2 = min(x2, width - 1) + BBOX_OFFSET
    y2 = min(y2, height - 1) + BBOX_OFFSET

    return {
        'name': name, 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0',
        'bndbox': {'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2}
    }

def convertCOCOtoVOC(cats, imags, anns, dst_base, classRenameDict):
    #dst_base = os.path.join("data", "VOCdevkitCOCO", "VOCCOCO")

    dst_dirs = {x: os.path.join(dst_base, x) for x in ["Annotations", "ImageSets", "JPEGImages"]}
    dst_dirs['ImageSets'] = os.path.join(dst_dirs['ImageSets'], "Main")
    # for k, d in dst_dirs.items():
    #     os.makedirs(d, exist_ok=True)

    #cate = {x['id']: x['name'] for x in json.load(open(sets["test"]))['categories']}
    cate = {x['id']: x['name'] for x in cats}


    images = {}
    for im in tqdm(imags, "Parse Images"):
        img = base_dict(im['coco_url'], im['width'], im['height'], 3)
        images[im["id"]] = img

    for an in tqdm(anns, "Parse Annotations"):
        catName = cate[an['category_id']]
        catName = catName.replace('_', ' ')
        renamedCatName = classRenameDict[catName]

        ann = base_object(images[an['image_id']]['annotation']["size"], renamedCatName, an['bbox'])
        images[an['image_id']]['annotation']['object'].append(ann)

    for k, im in tqdm(images.items(), "Write Annotations"):
        im['annotation']['object'] = im['annotation']['object'] or [None]
        unparse(im,
                open(os.path.join(dst_dirs["Annotations"], "{}.xml".format(str(k).zfill(12))), "w"),
                full_document=False, pretty=True)

    # print("Write image sets")
    # with open(os.path.join(dst_dirs["ImageSets"], "{}.txt".format(stage)), "w") as f:
    #     f.writelines(list(map(lambda x: str(x).zfill(12) + "\n", images.keys())))



def getLVISbyCategories(chosenSet, selectedCats, dstFolder, classRenameDict, limit):

    validSets = {'train', 'val'}
    dataUrls = {'train': "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip", 
                'val': "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip"}
    dataFiles = {'train': 'lvis_v1_train.json',
                'val': 'lvis_v1_val.json'}

    if chosenSet not in validSets:
        raise ValueError("getLVISbyCategories: chosenSet must be one of %r." % validSets)

    # Download data if not already downloadet
    dataPath = os.path.join(dstFolder, dataFiles[chosenSet])

    if not os.path.isfile(os.path.join(dstFolder, dataFiles[chosenSet])):
        print("Downloading LVIS {} annotations...".format(chosenSet))
        zipPath = dataPath + ".zip"
        wget.download(dataUrls[chosenSet], zipPath)

        # Unzip the data
        with zipfile.ZipFile(zipPath, 'r') as zip_ref:
            zip_ref.extractall(dstFolder)

        os.remove(zipPath)

    dst_dirs = {x: os.path.join(dstFolder, x) for x in ["Annotations", "ImageSets", "JPEGImages"]}
    dst_dirs['ImageSets'] = os.path.join(dst_dirs['ImageSets'], "Main")
    for k, d in dst_dirs.items():
        os.makedirs(d, exist_ok=True)

    anns = lvis.LVIS(dataPath)

    catids = anns.get_cat_ids()

    allcats = anns.load_cats(catids)

    catIndices = dict()
    catlist = []
    selectedCatInfo = []

    for cat in allcats:
        catIndices[cat['name'].lower()] = cat['id']
        catlist.append(cat['name'].lower())

    extractCats = []
    extractIndices = []

    for scat in selectedCats:
        cat = scat.replace('\n','')
        cat = cat.replace(' ', '_').lower()
        matches = [c for c in catlist if cat == c]

        if len(matches) == 0:
            # Try a more gentle approach to take missing words into account
            matches = [c for c in catlist if cat in c]
            
            print("Could not find match in LVIS database for {}".format(cat))
            print("Closest matches: {}".format(matches))
            matches = []

        for match in matches:
            extractCats.append(match)
            extractIndices.append(catIndices[match])

    # Make a copy of the category list to include in the new
    # subset of the annotation file
    selectedCatInfo = anns.load_cats(extractIndices)

    # Get the annotation ids given the category ids that we have defined
    annIds = []

    for idx in extractIndices:
        tempAnnIds = anns.get_ann_ids(cat_ids=[idx])

        # Only get the first 'limit' ann ids
        if len(tempAnnIds) > limit:
            annIds.extend(tempAnnIds[:limit])
        else:
            annIds.extend(tempAnnIds)


    lvisanns = anns.load_anns(ids=annIds)
    imageids = set()

    for ann in lvisanns:
        imageids.add(ann['image_id'])

    # Download images
    for imageId in tqdm(imageids, "Downloading LVIS {} images...".format(chosenSet)):
        try:
            anns.download(os.path.join(dstFolder, 'JPEGImages'), 
                        [imageId])
        except ConnectionError as e:
            print("Connection hiccup, trying again in 1 sec...")
            time.sleep(1)
            anns.download(os.path.join(dstFolder, 'JPEGImages'), 
                        [imageId])


    selectedAnnotations = dict()
    selectedAnnotations['categories'] = selectedCatInfo

    selectedAnnotations['images'] = anns.load_imgs(imageids)
    selectedAnnotations['annotations'] = lvisanns

    convertCOCOtoVOC(selectedCatInfo, 
                    selectedAnnotations['images'], 
                    lvisanns,
                    dstFolder, 
                    classRenameDict)

#getLVISbyCategories('val', ['boat', 'airplane'], './datasets', dict(), 300)

# COCO to VOC converter:
# https://gist.github.com/jinyu121/a222492405890ce912e95d8fb5367977
import lvis
import json
import os

from tqdm import tqdm
from xmltodict import unparse

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

def convertCOCOtoVOC(stage, cats, imags, anns, dst_base):
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
        ann = base_object(images[an['image_id']]['annotation']["size"], cate[an['category_id']], an['bbox'])
        images[an['image_id']]['annotation']['object'].append(ann)

    for k, im in tqdm(images.items(), "Write Annotations"):
        im['annotation']['object'] = im['annotation']['object'] or [None]
        unparse(im,
                open(os.path.join(dst_dirs["Annotations"], "{}.xml".format(str(k).zfill(12))), "w"),
                full_document=False, pretty=True)

    print("Write image sets")
    with open(os.path.join(dst_dirs["ImageSets"], "{}.txt".format(stage)), "w") as f:
        f.writelines(list(map(lambda x: str(x).zfill(12) + "\n", images.keys())))

    print("OK")



def getLVISbyCategories(set, selectedCats, dstFolder):

    dst_dirs = {x: os.path.join(dstFolder, x) for x in ["Annotations", "ImageSets", "JPEGImages"]}
    dst_dirs['ImageSets'] = os.path.join(dst_dirs['ImageSets'], "Main")
    for k, d in dst_dirs.items():
        os.makedirs(d, exist_ok=True)

    # WGet the LVIS annotations here to subfolder
    # Goes in val/train variants, solve this with argparse
    anns = lvis.LVIS('lvis_v1_val.json')

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
        matches = [c for c in catlist if cat in c]

        if len(matches) == 0:
            print("Could not find match for {}".format(cat))

        for match in matches:
            extractCats.append(match)
            extractIndices.append(catIndices[match])

    # Make a copy of the category list to include in the new
    # subset of the annotation file
    selectedCatInfo = anns.load_cats(extractIndices)

    # Get the annotation ids given the category ids that we have defined
    annids = anns.get_ann_ids(cat_ids=extractIndices)

    lvisanns = anns.load_anns(ids=annids)
    imageids = set()

    for ann in lvisanns:
        imageids.add(ann['image_id'])

    # Download images
    # for imageId in tqdm(imageids, "Downloading images..."):
    #     anns.download('JPEGImages', imageId)

    selectedAnnotations = dict()
    selectedAnnotations['categories'] = selectedCatInfo

    selectedAnnotations['images'] = anns.load_imgs(imageids)
    selectedAnnotations['annotations'] = lvisanns

    # with open('selectedAnns.json', 'w') as f:
    #     json.dump(selectedAnnotations, f)

    convertCOCOtoVOC('test', 
                    selectedCatInfo, 
                    selectedAnnotations['images'], 
                    lvisanns)

# To write in new annotation file
# categories
# annotations -> image_id, size, category_id, bbox
# images -> coco_url, width, height

# COCO to VOC converter:
# https://gist.github.com/jinyu121/a222492405890ce912e95d8fb5367977



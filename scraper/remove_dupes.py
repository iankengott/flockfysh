from imutils import paths
import numpy as np
import argparse
import cv2
import os


def dhash(image, hashSize=8):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	resized = cv2.resize(gray, (hashSize + 1, hashSize))
	diff = resized[:, 1:] > resized[:, :-1]
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def remove(dataset='', remove=1):
    total_removed = 0
    imagePaths = list(paths.list_images(dataset))
    hashes = {}
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        h = dhash(image)
        p = hashes.get(h, [])
        p.append(imagePath)
        hashes[h] = p


    for (h, hashedPaths) in hashes.items():
        if len(hashedPaths) > 1:
            if remove <= 0:
                montage = None
                for p in hashedPaths:
                    image = cv2.imread(p)
                    image = cv2.resize(image, (150, 150))
                    if montage is None:
                        montage = image
                    else:
                        montage = np.hstack([montage, image])
                print("[INFO] hash: {}".format(h))
                cv2.imshow("Montage", montage)
                cv2.waitKey(0)
            else:
                for p in hashedPaths[1:]:
                    os.remove(p)
                total_removed += len(hashedPaths[1:])
    return total_removed


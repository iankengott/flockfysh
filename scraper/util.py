import requests
import shutil
import posixpath
import urllib
import os
import imghdr
import validators
import base64
import uuid

from PIL import Image

DEFAULT_OUTPUT_DIR = "images"


def valid_image(file_path: str) -> None:
    file_path = file_path
    raw_name, ext = os.path.splitext(file_path)
    img_type = imghdr.what(file_path) or "jpeg"
    if f".{img_type}" != ext:
        new_file_path = f"{raw_name}.{img_type}"
        shutil.move(file_path, new_file_path)
        file_path = new_file_path
    try:
        img = Image.open(file_path)
        width, height = img.size
        if width == 0 or height == 0:
            os.remove(file_path)
        else:
            img = img.convert("RGB")
            img.save(file_path)
    except:
        os.remove(file_path)


def get_uuid() -> str:
    return str(uuid.uuid4().hex)


def download_image(url, path) -> bool:
    try:
        r = requests.get(url, stream=True, timeout=10, verify=False)
        if r.ok:
            ext = r.headers['Content-Type'].split("/")[-1].strip()
            filename = os.path.join(path, f'{get_uuid()}.{ext}')
            with open(filename, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
                # f.write(r.content)
            valid_image(filename)
        else:
            return False
    except Exception as e:
        return False 


def get_file_name(url, index, prefix='image') -> str:
    try:
        path = urllib.parse.urlsplit(url).path
        filename = posixpath.basename(path).split('?')[0]
        type, _ = file_data(filename)
        result = "{}_{}.{}".format(prefix, index, type)
        return result
    except Exception as e:
        print("[!] Get file name: {}\n[!] Err :: {}".format(url, e))
        return prefix


def rename(name, index, prefix='image') -> str:
    try:
        type, _ = file_data(name)
        result = "{}_{}.{}".format(prefix, index, type)
        return result
    except Exception as e:
        print("[!] Rename: {}\n[!] Err :: {}".format(name, e))
        return prefix


def file_data(name):
    try:
        type = name.split(".")[-1]
        name = name.split(".")[0]
        if type.lower() not in ["jpe", "jpeg", "jfif", "exif", "tiff", "gif", "bmp", "png", "webp", "jpg"]:
            type = "jpg"
        return (type, name)
    except Exception as e:
        print("[!] Issue getting: {}\n[!] Err :: {}".format(name, e))
        return (name, "jpg")


def make_image_dir(output_dir, force_replace=False) -> str:
    image_dir = output_dir
    if len(output_dir) < 1:
        image_dir = os.path.join(os.getcwd(), DEFAULT_OUTPUT_DIR)

    if force_replace:
        if os.path.isdir(image_dir):
            shutil.rmtree(image_dir)
    try:
        if not os.path.isdir(image_dir):
            os.makedirs(image_dir)
    except:
        pass

    return image_dir


if __name__ == '__main__':
    print("util")
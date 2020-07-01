# -*- coding: utf-8 -*-
import os
import sys
import zipfile
import glob


def unzip(filename):
    with zipfile.ZipFile(filename, "r") as zf:
        zf.extractall(path=os.path.dirname(filename))
    delete_zip(filename)


def delete_zip(zip_file):
    os.remove(zip_file)


def walk_in_dir(dir_path):
    for filename in glob.glob(os.path.join(dir_path, "*.zip")):
        unzip(filename=os.path.join(dir_path,filename))

    for dirname in (d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))):
        walk_in_dir(os.path.join(dir_path, dirname))


if __name__ == "__main__":
    args = sys.argv
    try:
        if(os.path.isdir(args[1])):
            walk_in_dir(args[1])
        else:
            unzip(os.path.join(args[1]))
            name, _ = os.path.splitext(args[1])
            if (os.path.isdir(name)):
                walk_in_dir(name)
    except IndexError:
        print('IndexError: Usage "python %s ZIPFILE_NAME" or "python %s DIR_NAME"' % (args[0], args[0]))
    except IOError:
        print('IOError: Couldn\'t open "%s"' % args[1])
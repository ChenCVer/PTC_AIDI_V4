import os
import csv
import codecs
import collections

import json
from ruamel import yaml


class FileHelper(object):
    """
    Provide some common file operation, includes create dictionary and file, read and write ...
    """

    @staticmethod
    def make_dirs(dir_path, is_file=False, debug=False):
        """
        Create a specified directory, dir_path is a file,
        this function will extract the last layer to create.
        If directory exits, no new directory will be created.
        Notice, this function can only create directory.

        :param debug:
        :param dir_path: string: specified path.
        :param is_file: bool: if True, given dir_path is a file, extracting directory will be conducted,
                              else, create dictionary directly.
        """
        dir_path = os.path.expanduser(dir_path)
        dir_name = FileHelper.get_dir_name(dir_path) if is_file else dir_path
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            if debug:
                print('Create path: ', dir_name)
        else:
            if debug:
                print('Path exists!')

    @staticmethod
    def get_dir_name(file_path):
        """
        Returns the directory component of a file path.
        """
        return os.path.dirname(file_path)

    @staticmethod
    def get_file_name(file_path):
        """
        Returns the file name of a file path.
        """
        FileHelper.check_file_exist(file_path)
        file_name = file_path.split('/')[-1]
        return file_name

    @staticmethod
    def get_abs_path(file_path):
        """
        Return the absolute version of a path.
        """
        return os.path.abspath(file_path)

    @staticmethod
    def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
        """
        Test whether a path is a regular file.
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError(msg_tmpl.format(filename))

    @staticmethod
    def check_path_exist(path, msg_tmpl='path "{}" does not exist'):
        """
        Test whether path exists.
        """
        if not os.path.exists(path):
            raise Exception(msg_tmpl.format(path))

    @staticmethod
    def get_file_path_list(path, extensions, exclude_extensions=None):
        """
        Recursively reads all files under given folder, until all files have been ergodic.
        You can also specified file extensions to read or not to read.
        :return: list: path_list contains all wanted files.
        """

        def is_valid_file(x):
            if exclude_extensions is None:
                return x.lower().endswith(extensions)
            else:
                return x.lower().endswith(extensions) and not x.lower().endswith(exclude_extensions)

        FileHelper.check_path_exist(path)
        if isinstance(extensions, list):
            extensions = tuple(extensions)
        if isinstance(exclude_extensions, list):
            exclude_extensions = tuple(exclude_extensions)

        all_list = os.listdir(path)
        path_list = []
        for subpath in all_list:
            path_next = os.path.join(path, subpath)
            if os.path.isdir(path_next):
                path_list.extend(FileHelper.get_file_path_list(path_next, extensions, exclude_extensions))
            else:
                if is_valid_file(path_next):
                    path_list.append(path_next)
        return path_list

    @staticmethod
    def is_dir(x):
        """
        Return true if the pathname refers to an existing directory.
        """
        return os.path.isdir(x)

    @staticmethod
    def make_folders(src, tar):
        """
        This function creates same directory structure (not files) as source in target directory.

        :param src: string: source directory
        :param tar: string: target directory
        """
        if not os.path.exists(tar):
            FileHelper.make_dirs(tar)
        paths = os.listdir(src)
        paths = map(lambda name: os.path.join(src, name), paths)
        paths = list(filter(FileHelper.is_dir, paths))
        if len(paths) <= 0:
            return
        for i in paths:
            _, filename = os.path.split(i)
            targetpath = os.path.join(tar, filename)
            if not FileHelper.is_dir(targetpath) and os.mkdir(targetpath):
                FileHelper.make_folders(i, targetpath)

    @staticmethod
    def read_csv(path):
        """
        The returned object is an iterator.  Each iteration returns a row
        of the CSV file (which can span multiple input lines).

        :param path: string: .csv file

        :return: list
        """
        FileHelper.check_file_exist(path)
        output_list = []
        if os.path.exists(path):
            with open(path, "r") as csvfile:
                reader = csv.reader(csvfile)
                for line in reader:
                    output_list.append(line)

        return output_list if len(output_list) > 0 else None

    @staticmethod
    def write_csv(path, datas):
        """
        Write datas in a given csv file.
        """
        file_csv = codecs.open(path, 'w+', 'utf-8')  # 追加
        writer = csv.writer(file_csv, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        for data in datas:
            writer.writerow(data)

    @staticmethod
    def read_yaml(path):
        """
        Parse the first YAML document in a stream
        and produce the corresponding Python object.
        """
        FileHelper.check_file_exist(path)
        a = open(path, 'r')
        datas = yaml.load(a.read(), Loader=yaml.Loader)

        return datas

    @staticmethod
    def write_yaml(path, datas):
        """
        Serialize a Python object into a YAML stream.

        """
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(datas, f, Dumper=yaml.Dumper)

    @staticmethod
    def read_json(path):
        """
        Support file-like object containing a JSON document to a Python object.

        """
        FileHelper.check_file_exist(path)
        with open(path, 'r') as f:
            datas = json.load(f)

        return datas

    @staticmethod
    def write_json(path, datas):
        """
        Serialize data as a JSON formatted stream to a file-like object.

        """
        with open(path, 'w') as f:
            json.dump(datas, f)

    @staticmethod
    def get_root_loader(path, keywords='0.bg'):
        """
        Get which directory level contains the specified  directory or file.

        :param path: string: raw_path.
        :param keywords: string: which you wanted.

        :return: list: a list that contains wanted directory or file.
        """
        FileHelper.check_path_exist(path)
        all_list = os.listdir(path)
        root_path = []
        for subpath in all_list:
            if subpath.find(keywords) >= 0:
                root_path = [path]
                return root_path
        for subpath in all_list:
            current_path = os.path.join(path, subpath)
            if os.path.isdir(current_path):
                root_path.append(FileHelper.get_root_loader(current_path, keywords))
        root_path = FileHelper.flatten(root_path)
        return root_path

    @staticmethod
    def flatten(x):
        """
        Flatten a list as length equals 1.

        :param x: list: element in list should be string, length of list varies.

        :return: list: a list whose length equals 1.
        """
        result = []
        for element in x:
            if isinstance(x, collections.Iterable) and not isinstance(element, str):
                result.extend(FileHelper.flatten(element))
            else:
                result.append(element)

        return result

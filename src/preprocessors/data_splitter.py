import xml.etree.ElementTree
import random

class DataSplitter:
    def __init__(self, data_file_path, train_percentage):
        self.data_file_path = data_file_path
        self.train_percentage = train_percentage

    def split_data(self, train_dest_path, test_dest_path):
        with open(self.data_file_path, 'rb') as xml_file:
            et = xml.etree.ElementTree.parse(xml_file)

            nodes = et.findall("sample")
            indexes = self.get_train_test_indexes(len(nodes))

        with open(self.data_file_path, 'rb') as xml_file:
            self.make_dataset(xml_file, indexes["train"], train_dest_path)
        with open(self.data_file_path, 'rb') as xml_file:
            self.make_dataset(xml_file, indexes["test"], test_dest_path)

    def make_dataset(self, file, indexes, dest):
        et = xml.etree.ElementTree.parse(file)
        root = et.getroot()

        for i, node in enumerate(root):
            if not self.sorted_contains(i, indexes):
                root.remove(node)

        et.write(dest)

    def get_train_test_indexes(self, length):
        result = {
            "test": [],
            "train": []
        }

        for i in range(length):
            r = random.random()

            if (r < self.train_percentage):
                result["test"].append(i)
            else:
                result["train"].append(i)

        return result

    def sorted_contains(self, index, arr):
        for el in arr:
            if el == index:
                return True
            elif el > index:
                return False
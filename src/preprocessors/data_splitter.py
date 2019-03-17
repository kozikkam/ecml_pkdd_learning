import xml.etree.ElementTree
import random


class DataSplitter:
    def __init__(self, data_file_path: str, train_percentage):
        self.data_file_path = data_file_path
        self.train_percentage = float(train_percentage)
        self.node_name = "sample";

    def split_data(self, train_dest_path, test_dest_path):
        with open(self.data_file_path, 'rb') as xml_file:
            et = xml.etree.ElementTree.parse(xml_file)

            nodes = et.findall(self.node_name)
            indexes = self.get_train_test_indexes(len(nodes))

        with open(self.data_file_path, 'rb') as xml_file:
            self.make_dataset(xml_file, indexes["train"], train_dest_path)
        with open(self.data_file_path, 'rb') as xml_file:
            self.make_dataset(xml_file, indexes["test"], test_dest_path)

    def make_dataset(self, file, indexes, dest):
        et = xml.etree.ElementTree.parse(file)
        root = et.getroot()
        nodes = et.findall(self.node_name)

        for i, node in enumerate(nodes):
            if i not in indexes:
                root.remove(node)

        et.write(dest)

    def get_train_test_indexes(self, length):
        result = {
            "test": [],
            "train": []
        }

        for i in range(length):
            r = random.random()

            if r < self.train_percentage:
                result["train"].append(i)
            else:
                result["test"].append(i)

        return result


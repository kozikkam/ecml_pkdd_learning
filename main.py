import os
import json
import fire
import logging

from src.preprocessors.xml_data_generator import XmlDataGenerator
from src.preprocessors.data_splitter import DataSplitter

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')


class PkddLearning:
    """
    Unsupervised SQL injection detection
    """

    def generate_data(self,
                      data_train_path: str = './dataset/data_train.xml',
                      data_test_path: str = './dataset/data_test.xml'):
        data_train_path = os.path.abspath(data_train_path)
        data_test_path = os.path.abspath(data_test_path)
        train_generator = XmlDataGenerator(data_train_path)
        test_generator = XmlDataGenerator(data_test_path)
        train_samples = iter(train_generator)
        test_samples = iter(test_generator)
        print(next(train_samples))
        print(next(train_samples))

    """
    Splits learning dataset into test and train datasets
    :param str data_file_path: Path to learning dataset
    :param str data_train_path: Path to train dataset (output)
    :param str data_test_path: Path to test dataset (output)
    :param float train_percentage: Set ratio of train to test data (a ratio of 0 means all data is test)
    """

    def split_data(self,
                   data_file_path: str = './dataset/learning_dataset.xml',
                   data_train_path: str = './dataset/data_train.xml',
                   data_test_path: str = './dataset/data_test.xml',
                   train_percentage: float = 0.7):
        logging.info(
            f"Begin splitting data. Source: {data_file_path}, train percentage: {train_percentage}")
        data_splitter = DataSplitter(data_file_path, train_percentage)
        data_splitter.split_data(data_train_path, data_test_path)


if __name__ == "__main__":
    fire.Fire(PkddLearning)

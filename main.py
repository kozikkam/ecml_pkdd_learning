import os
import json
import fire
import logging
import numpy as np

from src.preprocessors import DataSplitter
from src.generators import XmlDataGenerator
from src.extractors import FeatureExtractor

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')


class PkddLearning:
    """
    Unsupervised SQL injection detection
    """

    def extract_features(self,
                      cache_path: str = './features.dat',
                      data_train_path: str = './dataset/data_train.xml',
                      data_test_path: str = './dataset/data_test.xml'):
        data_train_path = os.path.abspath(data_train_path)
        data_test_path = os.path.abspath(data_test_path)
        train_generator = XmlDataGenerator(data_train_path)
        test_generator = XmlDataGenerator(data_test_path)
        features, labels = FeatureExtractor().extract(train_generator)
        np.savez(cache_path, np.array(features), np.array(labels))

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

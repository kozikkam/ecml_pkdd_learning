import os
import json

from src.preprocessors.xml_data_generator import XmlDataGenerator
from src.preprocessors.data_splitter import DataSplitter

data_path = os.path.abspath('./dataset/learning_dataset.xml')
data_train_path = os.path.abspath('./dataset/data_train.xml')
data_test_path = os.path.abspath('./dataset/data_test.xml')

data_splitter = DataSplitter(data_path, 0.7)
data_splitter.split_data(data_train_path, data_test_path)

train_generator = XmlDataGenerator(data_train_path)
test_generator = XmlDataGenerator(data_test_path)

train_samples = train_generator.get_generator()
test_samples = test_generator.get_generator()

print(next(train_samples))
print(next(train_samples))
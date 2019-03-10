import xml.etree.ElementTree as ET
import xmltodict

class XmlDataGenerator:
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path

    def __iter__(self):
        return self.get_generator()

    def get_generator(self):
        events = ET.iterparse(self.data_file_path, events=("start", "end",))

        for event, elem in events:
            if elem.tag == "sample" and event == "end":
                xmlstr = ET.tostring(elem).decode()
                data = xmltodict.parse(xmlstr)["sample"]

                yield data
                elem.clear()
import xml.etree.ElementTree as ET
import xmltodict
from .data_generator import DataGenerator


class XmlDataGenerator(DataGenerator):
    def _get_iterator(self):
        events = ET.iterparse(self._data_file_path, events=("start", "end",))

        for event, elem in events:
            if elem.tag == "sample" and event == "end":
                xmlstr = ET.tostring(elem).decode()
                data = xmltodict.parse(xmlstr)["sample"]

                yield data
                elem.clear()

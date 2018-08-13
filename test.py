import xml.etree.ElementTree as ET

tree = ET.parse('data/trollface_landmarks.xml')
root = tree.getroot()

for p in root.iter('part'):
    p.attrib['x'] = str(int(p.attrib['x']) + 1)

tree.write('data/trollface_landmarks.xml')
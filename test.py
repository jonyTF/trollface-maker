import xml.etree.ElementTree as ET

tree = ET.parse('data/trollface_landmarks.xml')
root = tree.getroot()

for p in root.iter('part'):
    p.attrib['y'] = str(int(p.attrib['y']) - 52)

tree.write('trollface_landmarks_newer.xml')
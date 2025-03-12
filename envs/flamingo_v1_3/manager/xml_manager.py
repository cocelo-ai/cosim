import numpy as np
import xml.etree.ElementTree as ET
import os

class XMLManager:
    def __init__(self, config):
        self.config = config
        self.cur_dir = os.path.abspath(os.path.dirname(__file__))
        self.body_components =["base_link", "left_hip_link", "right_hip_link", "left_shoulder_link", "right_shoulder_link",
                          "left_leg_link", "right_leg_link", "left_wheel_link", "right_wheel_link"]

        self.precision_attr_map = config["random_table"]["precision"]

    def get_model_path(self):
        original_model_path = os.path.join(self.cur_dir, '..', 'assets', 'xml', 'flamingo_v1_3.xml')
        tree = ET.parse(original_model_path)
        root = tree.getroot()

        # 1. Set the terrain
        terrain = self.config["env"]["terrain"]
        for geom in root.findall('.//geom'):
            if geom.attrib.get('name') == "ground":
                geom.attrib["hfield"] = terrain

        # 2. Set the precision of the simulation
        precision_level = self.config["random"]["precision"]
        if precision_level in self.precision_attr_map:
            precision_attrs = self.precision_attr_map[precision_level]
            option = root.find("option")
            if option is not None:
                option.attrib["timestep"] = str(precision_attrs["timestep"])
                option.attrib["iterations"] = str(precision_attrs["iterations"])

        # 3. Set the noisy mass & load
        for body in root.findall('.//body'):
            body_name = body.attrib.get('name')
            if body_name in self.body_components:
                for inertial in body.findall('inertial'):
                    if 'mass' in inertial.attrib:
                        original_mass = float(inertial.attrib['mass'])
                        noise = np.random.uniform(-original_mass * self.config["random"]["mass_noise"],
                                                  original_mass * self.config["random"]["mass_noise"])
                        randomized_mass = original_mass + noise
                        if body_name == "base_link":
                            randomized_mass += self.config["random"]["load"]
                        inertial.attrib['mass'] = str(randomized_mass)

        # 4. Set the friction of wheel geoms in left_wheel_link and right_wheel_link
        for body in root.findall('.//body'):
            if body.attrib.get('name') in ['left_wheel_link', 'right_wheel_link']:
                for geom in body.findall('geom'):
                    if 'friction' in geom.attrib:
                        geom.attrib['friction'] = (
                            f"{self.config['random']['sliding_friction']} "
                            f"{self.config['random']['torsional_friction']} "
                            f"{self.config['random']['rolling_friction']}"
                        )

        # 5. Set the friction of ground plane
        for geom in root.findall('.//geom'):
            geom_name = geom.attrib.get('name')
            if geom_name == "ground":
                if 'friction' in geom.attrib:
                    geom.attrib['friction'] = (str(self.config["random"]["sliding_friction"])
                                               + ' ' + str(self.config["random"]["torsional_friction"])
                                               + ' ' + str(self.config["random"]["rolling_friction"]))

        # 6. Set the frictionloss
        for default in root.findall(".//default"):
            default_class = default.attrib.get("class")
            if default_class == "joints":
                for joint in default.findall("joint"):
                    if 'frictionloss' in joint.attrib:
                        joint.attrib['frictionloss'] = str(self.config["random"]["friction_loss"])
            elif default_class == "wheels":
                for joint in default.findall("joint"):
                    if 'frictionloss' in joint.attrib:
                        joint.attrib['frictionloss'] = str(self.config["random"]["friction_loss"])


        randomized_model_path = os.path.join(self.cur_dir, '..', 'assets', 'xml', 'applied_flamingo_v1_3.xml')
        tree.write(randomized_model_path)
        return randomized_model_path


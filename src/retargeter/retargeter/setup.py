from setuptools import setup, find_packages

package_name = "retargeter"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["examples"]),
    install_requires=["setuptools", "scipy", "pytorch_kinematics", "torch"],
    zip_safe=True,
    maintainer="Chenyu Yang",
    maintainer_email="chenyu.yang@srl.ethz.ch",
    description="Python package for MANO retargeting",
    license="MIT",
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/retargeter"]),
        ("share/retargeter", ["package.xml"]),
    ],
)

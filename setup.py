from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ip'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        # (os.path.join('share', package_name, 'config', 'ur5e'), glob('config/ur5e/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='ROS version of IP policy package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'eval_pose_control = ip.eval_pose_control:main',
            'eval_cartesian_motion = ip.eval_cartesian_motion:main',
        ],
    },
)
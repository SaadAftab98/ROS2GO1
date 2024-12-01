from setuptools import setup

package_name = 'go1_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools', 'rclpy', 'opencv-python', 'cv_bridge', 'ultralytics'],
    entry_points={
        'console_scripts': [
            'yolov8_number_detection_node = go1_detection.yolov8_number_detection_node:main',  # Update this line
        ],
    },
)


from setuptools import find_packages, setup

package_name = 'oak_vlm_pipeline'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shshsh',
    maintainer_email='shshsh@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'rgb_depth_sync_node = oak_vlm_pipeline.rgb_depth_sync_node:main',
        'bbox_depth_xyz_node = oak_vlm_pipeline.bbox_depth_xyz_node:main',
        'qwen_bbox_visualizer = oak_vlm_pipeline.qwen_bbox_visualizer:main',
        'qwen7b_vlm_node = oak_vlm_pipeline.qwen7b_vlm_node:main',
        ],
    },
)

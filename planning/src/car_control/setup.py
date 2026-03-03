from setuptools import find_packages, setup

package_name = 'car_control'

setup(
    name=package_name, # 專案名稱
    version='0.0.0',    # 專案版本
    packages=find_packages(exclude=['test']), 
    data_files=[    # 定義執行檔所需要的路徑
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files
        ('share/' + package_name + '/launch', ['launch/car_sensor.launch.py']),
        # yaml config files
        ('share/' + package_name + '/config', ['config/car_controller.yaml']),
        # rviz config files
        ('share/' + package_name + '/config', ['config/car_odom.rviz']),
    ],  
    install_requires=['setuptools','pyserial', 'tf-transformations'],   # 定義主節點所宣告的套件
    zip_safe=True,
    maintainer='zanrobot',  # 專案維護者
    maintainer_email='zanrobot@todo.todo',  # 專案維護者的電子郵件
    description='TODO: Package description',    # 專案描述
    license='TODO: License declaration', # 專案授權
    tests_require=['pytest'],
    entry_points={  
        'console_scripts': [    # 定義主程式節點
            # 呼叫時的名稱 = 套件名稱.主程式名稱:主函式
            'car_controller_node = car_control.car_controller:main',
        ],
    },
)

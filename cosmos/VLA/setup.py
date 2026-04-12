from setuptools import find_packages, setup


package_name = "vla_demo"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "tests"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", [
            "launch/mission_stack.launch.py",
            "launch/reasoner.launch.py",
            "launch/executor.launch.py",
        ]),
        ("share/" + package_name + "/config", ["config/sample_mission.yaml"]),
    ],
    install_requires=["setuptools", "PyYAML"],
    zip_safe=True,
    maintainer="Codex",
    maintainer_email="codex@openai.com",
    description="ROS2 AMR mission orchestration and Cosmos-Reason2 integration package.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "vla_mission_manager = vla_demo.mission_manager:main",
            "cosmos_reasoner_node = vla_demo.reasoner_node:main",
            "vla_executor = vla_demo.executor_node:main",
            "camera_front_node = vla_demo.camera_front_node:main",
        ],
    },
)

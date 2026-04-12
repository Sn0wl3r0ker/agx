cmake_minimum_required(VERSION 3.8)
project(car_control)

# ============================================================
#  C++ 標準
# ============================================================
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ============================================================
#  尋找依賴
# ============================================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)

# nlohmann/json — 優先嘗試 find_package，若無則以 header-only 方式 fetch
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
  message(STATUS "nlohmann_json not found via find_package, using FetchContent")
  include(FetchContent)
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
  )
  FetchContent_MakeAvailable(nlohmann_json)
endif()

# ============================================================
#  C++ 執行檔
# ============================================================

# --- serial_bridge_node ---
# 職責：獨佔 Serial Port，使用獨立 Reader Thread 做非阻塞讀取，
#       根據 JSON Key 分流發布到 /raw_odom, /battery_state, /charge_status
add_executable(serial_bridge_node src/serial_bridge_node.cpp)
ament_target_dependencies(serial_bridge_node
  rclcpp
  std_msgs
)

# --- kinematics_node ---
# 職責：cmd_vel→輪速、encoder→Odom/TF、電量/充電站狀態轉發
add_executable(kinematics_node src/kinematics_node.cpp)
ament_target_dependencies(kinematics_node
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  tf2_ros
  tf2
)
target_link_libraries(kinematics_node nlohmann_json::nlohmann_json)

# ============================================================
#  安裝 C++ 執行檔
# ============================================================
install(TARGETS
  serial_bridge_node
  kinematics_node
  DESTINATION lib/${PROJECT_NAME}
)

# ============================================================
#  安裝 Launch / Config / Resource 檔案
# ============================================================
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY urdf/
  DESTINATION share/${PROJECT_NAME}/urdf
)

install(FILES resource/${PROJECT_NAME}
  DESTINATION share/ament_index/resource_index/packages
)

install(FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

# ============================================================
#  安裝 Python 腳本（里程計評估工具）
# ============================================================
install(PROGRAMS
  scripts/odom_recorder_node.py
  scripts/plot_trajectory.py
  DESTINATION lib/${PROJECT_NAME}
)

# ============================================================
#  測試
# ============================================================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

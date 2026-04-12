/**
 * @file kinematics_node.cpp
 * @brief ROS 2 Kinematics Node — 處理 cmd_vel→輪速 與 encoder→Odom/TF，整合電量與充電站狀態
 *
 * 架構：
 *   ┌──────────────────────────────────────────────────────────────────────────┐
 *   │  kinematics_node                                                        │
 *   │                                                                          │
 *   │  [輸入]                                                                  │
 *   │    /cmd_vel  (Twist)          → 差速逆運動學 → /motor_cmd (JSON String)  │
 *   │    /raw_encoder_json (String) → 正運動學 → /raw_odom + TF (odom→base_link)   │
 *   │    /battery_state (String)    → 解析電壓 → /battery_voltage (Float32)    │
 *   │    /charge_status (String)    → 解析狀態 → /charging_state (String)      │
 *   │                                                                          │
 *   │  [重要] 時間戳使用「收到 JSON 瞬間的 now()」，確保 TF 精準度             │
 *   └──────────────────────────────────────────────────────────────────────────┘
 *
 * 運動學公式 (差速驅動)：
 *   正運動學 (cmd_vel → 輪速)：
 *     v_left  = linear_x - (angular_z × L / 2)
 *     v_right = linear_x + (angular_z × L / 2)
 *     RPM = (v / (π × D)) × 60 × gear_ratio
 *
 *   逆運動學 (encoder → Odom)：
 *     dist_L = Δp1 × (π × D / total_pulses)
 *     dist_R = Δp2 × (π × D / total_pulses)
 *     vx = (dist_R + dist_L) / (2 × dt)
 *     wz = (dist_R - dist_L) / (L × dt)
 *
 * 編譯依賴：rclcpp, std_msgs, geometry_msgs, nav_msgs, tf2_ros, tf2, nlohmann_json
 */

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float32.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <string>
#include <optional>
#include <sstream>

using json = nlohmann::json;

class KinematicsNode : public rclcpp::Node
{
public:
    KinematicsNode()
        : Node("kinematics_node")
    {
        // =====================================================================
        //  參數宣告與讀取
        // =====================================================================
        this->declare_parameter<double>("car_distance", 0.65);
        this->declare_parameter<std::string>("car_mode", "diff");
        this->declare_parameter<double>("Tire_diameter", 0.18);
        this->declare_parameter<int>("encoder_resolution", 720);
        this->declare_parameter<double>("gear_ratio", 1.0);
        this->declare_parameter<int>("odom_up_hz", 10);
        this->declare_parameter<bool>("use_encoder_feedback", true);
        this->declare_parameter<std::string>("odom_frame", "odom");
        this->declare_parameter<std::string>("base_frame", "base_link");
        this->declare_parameter<std::string>("base_footprint_frame", "base_footprint");

        car_distance_        = this->get_parameter("car_distance").as_double();
        car_mode_            = this->get_parameter("car_mode").as_string();
        tire_diameter_       = this->get_parameter("Tire_diameter").as_double();
        encoder_resolution_  = this->get_parameter("encoder_resolution").as_int();
        gear_ratio_          = this->get_parameter("gear_ratio").as_double();
        odom_up_hz_          = this->get_parameter("odom_up_hz").as_int();
        use_encoder_feedback_= this->get_parameter("use_encoder_feedback").as_bool();
        odom_frame_          = this->get_parameter("odom_frame").as_string();
        base_frame_          = this->get_parameter("base_frame").as_string();
        base_footprint_frame_= this->get_parameter("base_footprint_frame").as_string();

        // =====================================================================
        //  衍生常數計算
        // =====================================================================
        tire_circumference_ = M_PI * tire_diameter_;
        tire_radius_        = tire_diameter_ / 2.0;
        total_pulses_       = gear_ratio_ * static_cast<double>(encoder_resolution_);
        distance_per_pulse_ = M_PI * tire_diameter_ / total_pulses_;

        RCLCPP_INFO(this->get_logger(), "======== Kinematics 參數 ========");
        RCLCPP_INFO(this->get_logger(), "car_distance       = %.3f m", car_distance_);
        RCLCPP_INFO(this->get_logger(), "car_mode           = %s", car_mode_.c_str());
        RCLCPP_INFO(this->get_logger(), "Tire_diameter      = %.3f m", tire_diameter_);
        RCLCPP_INFO(this->get_logger(), "encoder_resolution = %d", encoder_resolution_);
        RCLCPP_INFO(this->get_logger(), "gear_ratio         = %.2f", gear_ratio_);
        RCLCPP_INFO(this->get_logger(), "tire_circumference = %.4f m", tire_circumference_);
        RCLCPP_INFO(this->get_logger(), "distance_per_pulse = %.6f m", distance_per_pulse_);
        RCLCPP_INFO(this->get_logger(), "odom_frame         = %s", odom_frame_.c_str());
        RCLCPP_INFO(this->get_logger(), "base_frame         = %s", base_frame_.c_str());
        RCLCPP_INFO(this->get_logger(), "base_footprint     = %s", base_footprint_frame_.c_str());
        RCLCPP_INFO(this->get_logger(), "use_encoder_feedback = %s",
                     use_encoder_feedback_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "=================================");

        // =====================================================================
        //  Odom 狀態初始化
        // =====================================================================
        x_ = 0.0;
        y_ = 0.0;
        theta_ = 0.0;
        last_linear_x_ = 0.0;
        last_linear_y_ = 0.0;
        last_angular_z_ = 0.0;
        last_time_ = this->now();

        // =====================================================================
        //  ROS 介面 — Publishers
        // =====================================================================
        // 馬達速度 JSON → serial_bridge_node
        pub_motor_cmd_      = this->create_publisher<std_msgs::msg::String>("motor_cmd", 10);
        // Odometry
        pub_odom_           = this->create_publisher<nav_msgs::msg::Odometry>("raw_odom", 30);
        // 電池電壓 (解析後的浮點數)
        pub_battery_voltage_= this->create_publisher<std_msgs::msg::Float32>("battery_voltage", 10);
        // 充電站狀態 (解析後的結構化 JSON 字串，供上層決策用)
        pub_charging_state_ = this->create_publisher<std_msgs::msg::String>("charging_state", 10);

        // =====================================================================
        //  ROS 介面 — Subscribers
        // =====================================================================
        // /cmd_vel → 差速運動學 → motor_cmd
        sub_cmd_vel_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "cmd_vel", 10,
            std::bind(&KinematicsNode::cmd_vel_callback, this, std::placeholders::_1));

        // /raw_encoder_json ← serial_bridge_node (已分流的 encoder JSON)
        sub_raw_odom_ = this->create_subscription<std_msgs::msg::String>(
            "raw_encoder_json", 30,
            std::bind(&KinematicsNode::raw_odom_callback, this, std::placeholders::_1));

        // /battery_state ← serial_bridge_node (已分流的電量 JSON)
        sub_battery_state_ = this->create_subscription<std_msgs::msg::String>(
            "battery_state", 10,
            std::bind(&KinematicsNode::battery_state_callback, this, std::placeholders::_1));

        // /charge_status ← serial_bridge_node (已分流的充電站 JSON)
        sub_charge_status_ = this->create_subscription<std_msgs::msg::String>(
            "charge_status", 10,
            std::bind(&KinematicsNode::charge_status_callback, this, std::placeholders::_1));

        // =====================================================================
        //  開迴路 / 閉迴路 模式切換
        // =====================================================================
        if (!use_encoder_feedback_) {
            const auto period = std::chrono::milliseconds(1000 / odom_up_hz_);
            odom_timer_ = this->create_wall_timer(
                period, std::bind(&KinematicsNode::odom_openloop_callback, this));
            RCLCPP_INFO(this->get_logger(), "開迴路模式：以 %d Hz 積分 Odom", odom_up_hz_);
        } else {
            RCLCPP_INFO(this->get_logger(),
                        "閉迴路模式：等待 Arduino Push {\"p1\":x,\"p2\":y}");
        }

        RCLCPP_INFO(this->get_logger(), "Kinematics Node 啟動完成");
    }

private:
    // =========================================================================
    //  /cmd_vel → 差速運動學 → /motor_cmd
    // =========================================================================

    /**
     * @brief /cmd_vel 回呼
     *
     * 差速運動學：
     *   v_left  = linear_x - (angular_z × L / 2)
     *   v_right = linear_x + (angular_z × L / 2)
     *
     * m/s → RPM：
     *   RPM = (v / circumference) × 60 × gear_ratio
     *
     * 輸出 JSON 格式：{"ls":<RPM>, "rs":<RPM>}
     * ⚠️ 注意：不再包含 "m":0，因為新版 Arduino (ros2_com.ino)
     *          只認 "ls" 和 "rs" 兩個 Key
     */
    void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        if (car_mode_ != "diff") {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                  "目前僅支援 diff (差速) 模式！");
            return;
        }

        const double linear_x  = msg->linear.x;   // m/s
        const double angular_z = msg->angular.z;   // rad/s

        // 暫存 (供開迴路 Odom 使用)
        last_linear_x_  = linear_x;
        last_linear_y_  = msg->linear.y;
        last_angular_z_ = angular_z;

        // ---- 差速運動學 ----
        const double v_left  = linear_x - (angular_z * car_distance_ / 2.0);
        const double v_right = linear_x + (angular_z * car_distance_ / 2.0);

        // ---- m/s → RPM ----
        const int rpm_left  = static_cast<int>(
            (v_left  / tire_circumference_) * 60.0 * gear_ratio_);
        const int rpm_right = static_cast<int>(
            (v_right / tire_circumference_) * 60.0 * gear_ratio_);

        // ---- 組合 JSON (使用 nlohmann/json) ----
        json cmd;
        cmd["ls"] = rpm_left;
        cmd["rs"] = rpm_right;

        auto tx_msg = std_msgs::msg::String();
        tx_msg.data = cmd.dump();
        pub_motor_cmd_->publish(tx_msg);

        RCLCPP_DEBUG(this->get_logger(), "cmd_vel → motor_cmd: %s", tx_msg.data.c_str());
    }

    // =========================================================================
    //  /raw_encoder_json → Encoder 解析 → Odom + TF (閉迴路)
    // =========================================================================

    /**
     * @brief /raw_encoder_json 回呼
     *
     * 收到 serial_bridge_node 分流後的 encoder JSON: {"p1":xxx,"p2":xxx}
     * 「立刻」使用 now() 取得時間戳，然後計算正運動學
     *
     * ⚠️ 關鍵：時間戳必須是「ROS 節點收到 JSON 的瞬間」，
     *         而不是 Arduino 發送的時間（我們無法得知），
     *         這樣 TF 的時間戳才能與其他感測器正確對齊。
     */
    void raw_odom_callback(const std_msgs::msg::String::SharedPtr msg)
    {
        if (!use_encoder_feedback_) return;

        // ⚠️ 在解析之前就取 now()，確保最精準的時間戳
        const auto stamp_now = this->now();

        try {
            auto j = json::parse(msg->data);

            if (j.contains("p1") && j.contains("p2")) {
                last_left_encoder_  = j["p1"].get<long>();
                last_right_encoder_ = j["p2"].get<long>();
                compute_encoder_odom(stamp_now);
            }
        } catch (const json::parse_error &) {
            RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                  "丟棄非 JSON 字串: %s", msg->data.c_str());
        }
    }

    /**
     * @brief 從 encoder ticks 計算 Odometry 並發布
     * @param stamp 使用「收到 JSON 瞬間」的時間戳
     */
    void compute_encoder_odom(const rclcpp::Time &stamp)
    {
        const double dt = (stamp - last_time_).seconds();
        if (dt <= 0.0) return;

        if (prev_left_encoder_.has_value() && prev_right_encoder_.has_value()) {
            const long delta_left  = last_left_encoder_  - prev_left_encoder_.value();
            const long delta_right = last_right_encoder_ - prev_right_encoder_.value();

            const double dist_left  = static_cast<double>(delta_left)  * distance_per_pulse_;
            const double dist_right = static_cast<double>(delta_right) * distance_per_pulse_;

            const double odom_vx = (dist_right + dist_left) / (2.0 * dt);
            const double odom_vy = 0.0;
            const double odom_wz = (dist_right - dist_left) / (car_distance_ * dt);

            integrate_odom(odom_vx, odom_vy, odom_wz, dt);
            publish_odom_and_tf(stamp, odom_vx, odom_vy, odom_wz);
        }

        prev_left_encoder_  = last_left_encoder_;
        prev_right_encoder_ = last_right_encoder_;
        last_time_ = stamp;
    }

    // =========================================================================
    //  電池與充電站狀態處理
    // =========================================================================

    /**
     * @brief /battery_state 回呼 — 解析 {"pow":24.5}
     *        發布解析後的電壓浮點數到 /battery_voltage (Float32)
     */
    void battery_state_callback(const std_msgs::msg::String::SharedPtr msg)
    {
        try {
            auto j = json::parse(msg->data);
            if (j.contains("pow")) {
                auto voltage_msg = std_msgs::msg::Float32();
                voltage_msg.data = j["pow"].get<float>();
                pub_battery_voltage_->publish(voltage_msg);
                RCLCPP_DEBUG(this->get_logger(), "電池電壓: %.1f V", voltage_msg.data);
            }
        } catch (const json::parse_error &) {
            // 忽略
        }
    }

    /**
     * @brief /charge_status 回呼 — 解析 {"can_v":0.00,"can_w":0.00,"c_st":0}
     *        直接轉發原始 JSON 到 /charging_state，讓上層決策節點使用
     */
    void charge_status_callback(const std_msgs::msg::String::SharedPtr msg)
    {
        try {
            auto j = json::parse(msg->data);
            if (j.contains("can_v") && j.contains("can_w") && j.contains("c_st")) {
                pub_charging_state_->publish(*msg);

                int c_st = j["c_st"].get<int>();
                if (c_st == 1) {
                    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                         "充電站回報：正在充電");
                }
            }
        } catch (const json::parse_error &) {
            // 忽略
        }
    }

    // =========================================================================
    //  開迴路 Odom (使用 cmd_vel 積分，無 encoder feedback)
    // =========================================================================

    void odom_openloop_callback()
    {
        const auto current_time = this->now();
        const double dt = (current_time - last_time_).seconds();
        if (dt <= 0.0) return;

        integrate_odom(last_linear_x_, last_linear_y_, last_angular_z_, dt);
        publish_odom_and_tf(current_time, last_linear_x_, last_linear_y_, last_angular_z_);

        last_time_ = current_time;
    }

    // =========================================================================
    //  共用：Odom 積分 & 發布
    // =========================================================================

    /**
     * @brief 2D Odom 積分
     *   dx = (vx × cos(θ) - vy × sin(θ)) × dt
     *   dy = (vx × sin(θ) + vy × cos(θ)) × dt
     *   dθ = wz × dt
     */
    void integrate_odom(double vx, double vy, double wz, double dt)
    {
        const double delta_x  = (vx * std::cos(theta_) - vy * std::sin(theta_)) * dt;
        const double delta_y  = (vx * std::sin(theta_) + vy * std::cos(theta_)) * dt;
        const double delta_th = wz * dt;

        x_     += delta_x;
        y_     += delta_y;
        theta_ += delta_th;
    }

    /**
     * @brief 發布 /raw_odom 訊息
     */
    void publish_odom_and_tf(const rclcpp::Time &stamp,
                             double vx, double vy, double wz)
    {
        // --- 四元數 ---
        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, theta_);

        // --- Odometry ---
        nav_msgs::msg::Odometry odom;
        odom.header.stamp    = stamp;
        odom.header.frame_id = odom_frame_;
        odom.child_frame_id  = base_footprint_frame_;

        odom.pose.pose.position.x = x_;
        odom.pose.pose.position.y = y_;
        odom.pose.pose.position.z = 0.0;
        odom.pose.pose.orientation.x = q.x();
        odom.pose.pose.orientation.y = q.y();
        odom.pose.pose.orientation.z = q.z();
        odom.pose.pose.orientation.w = q.w();

        odom.twist.twist.linear.x  = vx;
        odom.twist.twist.linear.y  = vy;
        odom.twist.twist.angular.z = wz;

        pub_odom_->publish(odom);
    }

    // =========================================================================
    //  成員變數
    // =========================================================================

    // --- 車體 / 運動學參數 ---
    double car_distance_;        // 輪距 (m)
    std::string car_mode_;       // "diff" 差速
    double tire_diameter_;       // 車輪直徑 (m)
    int encoder_resolution_;     // 編碼器每圈脈衝數
    double gear_ratio_;          // 減速比
    int odom_up_hz_;             // 開迴路 Odom 更新頻率
    bool use_encoder_feedback_;  // true=閉迴路 (Arduino Push), false=開迴路
    std::string odom_frame_;     // TF 父 frame
    std::string base_frame_;     // TF 子 frame
    std::string base_footprint_frame_; // TF 地面投影 frame

    double tire_circumference_;  // π × D
    double tire_radius_;         // D / 2
    double total_pulses_;        // gear_ratio × encoder_resolution
    double distance_per_pulse_;  // π × D / total_pulses

    // --- Odom 狀態 ---
    double x_, y_, theta_;
    double last_linear_x_, last_linear_y_, last_angular_z_;
    rclcpp::Time last_time_;

    // --- Encoder 追蹤 ---
    long last_left_encoder_{0};
    long last_right_encoder_{0};
    std::optional<long> prev_left_encoder_;
    std::optional<long> prev_right_encoder_;

    // --- ROS Publishers ---
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_motor_cmd_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pub_battery_voltage_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_charging_state_;

    // --- ROS Subscribers ---
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_cmd_vel_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_raw_odom_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_battery_state_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_charge_status_;

    // --- Timer (僅開迴路模式) ---
    rclcpp::TimerBase::SharedPtr odom_timer_;
};

// =============================================================================
//  main
// =============================================================================
int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<KinematicsNode>();
    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}

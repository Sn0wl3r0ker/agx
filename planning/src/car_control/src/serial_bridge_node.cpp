/**
 * @file serial_bridge_node.cpp
 * @brief ROS 2 Serial Bridge Node — 獨佔 Serial Port，負責 Arduino JSON ↔ ROS Topics 雙向轉換
 *
 * 架構：
 *   ┌────────────────────────────────────────────────────────────────────┐
 *   │  serial_bridge_node                                               │
 *   │                                                                    │
 *   │  [獨立 Reader Thread]                                              │
 *   │    │  POSIX read() + 行累積                                        │
 *   │    │  ↓ 逐行 JSON 解析                                             │
 *   │    ├─ {"p1":x,"p2":y}       → /raw_encoder_json       (std_msgs/String)   │
 *   │    ├─ {"pow":x}             → /battery_state   (std_msgs/String)   │
 *   │    ├─ {"can_v":x,...}       → /charge_status   (std_msgs/String)   │
 *   │    └─ 其他                  → /serial_rx        (std_msgs/String)   │
 *   │                                                                    │
 *   │  [ROS Callback]                                                    │
 *   │    /motor_cmd  (std_msgs/String) → UART TX  {"ls":x,"rs":y}\n     │
 *   │    /charge_cmd (std_msgs/String) → UART TX  {"charge":1}\n        │
 *   └────────────────────────────────────────────────────────────────────┘
 *
 * 設計決策：
 *   1. 使用獨立 std::thread 做 Serial 阻塞式讀取，絕對不會卡死 rclcpp::spin()
 *   2. 讀到完整 JSON 行後，根據 Key 值分發到對應的內部 Topic
 *   3. 節點關閉時自動送出停車命令 {"ls":0,"rs":0}
 *   4. Serial 使用 POSIX termios raw mode，8N1 (Jetson 側)
 *
 * 編譯依賴：rclcpp, std_msgs
 */

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <errno.h>
#include <poll.h>

#include <cstring>
#include <string>
#include <mutex>
#include <thread>
#include <atomic>

class SerialBridgeNode : public rclcpp::Node
{
public:
    SerialBridgeNode()
        : Node("serial_bridge_node"), serial_fd_(-1), running_(false)
    {
        // =====================================================================
        //  參數宣告
        // =====================================================================
        this->declare_parameter<std::string>("usb_port", "/dev/ttyACM0");
        this->declare_parameter<int>("baudrate", 115200);

        const auto port     = this->get_parameter("usb_port").as_string();
        const auto baudrate = this->get_parameter("baudrate").as_int();

        // =====================================================================
        //  開啟 Serial Port (POSIX termios, 8N1)
        // =====================================================================
        if (!open_serial(port, baudrate)) {
            RCLCPP_FATAL(this->get_logger(), "無法開啟 Serial Port: %s", port.c_str());
            throw std::runtime_error("Serial port open failed");
        }
        RCLCPP_INFO(this->get_logger(),
                     "成功連接 Arduino: %s @ %d baud", port.c_str(), static_cast<int>(baudrate));

        // =====================================================================
        //  Publishers — 依據 Arduino 推播的 JSON Key 分流
        // =====================================================================
        // 里程計 (20Hz): {"p1":x,"p2":y}
        pub_raw_odom_      = this->create_publisher<std_msgs::msg::String>("raw_encoder_json", 30);
        // 電池電量 (1Hz): {"pow":24.5}
        pub_battery_state_ = this->create_publisher<std_msgs::msg::String>("battery_state", 10);
        // 充電站狀態 (10Hz): {"can_v":x,"can_w":y,"c_st":0}
        pub_charge_status_ = this->create_publisher<std_msgs::msg::String>("charge_status", 10);
        // 其他未分類訊息 (系統訊息、debug 等)
        pub_serial_rx_     = this->create_publisher<std_msgs::msg::String>("serial_rx", 10);

        // =====================================================================
        //  Subscribers — ROS → Arduino
        // =====================================================================
        // 馬達速度命令: {"ls":RPM,"rs":RPM}
        sub_motor_cmd_ = this->create_subscription<std_msgs::msg::String>(
            "motor_cmd", 10,
            std::bind(&SerialBridgeNode::motor_cmd_callback, this, std::placeholders::_1));

        // 充電控制命令: {"charge":1} 或 {"charge":0}
        sub_charge_cmd_ = this->create_subscription<std_msgs::msg::String>(
            "charge_cmd", 10,
            std::bind(&SerialBridgeNode::charge_cmd_callback, this, std::placeholders::_1));

        // =====================================================================
        //  啟動獨立的 Serial Reader Thread
        // =====================================================================
        running_ = true;
        reader_thread_ = std::thread(&SerialBridgeNode::reader_thread_func, this);

        RCLCPP_INFO(this->get_logger(), "Serial Bridge 啟動完成 (Reader Thread 已啟動)");
    }

    ~SerialBridgeNode() override
    {
        // 通知 reader thread 結束
        running_ = false;
        if (reader_thread_.joinable()) {
            reader_thread_.join();
        }

        // 安全停車：關閉前送出零速命令
        if (serial_fd_ >= 0) {
            const std::string stop_cmd = "{\"ls\":0,\"rs\":0}\n";
            write_serial(stop_cmd);
            RCLCPP_INFO(this->get_logger(), "送出停車命令，關閉 Serial Port");
            ::close(serial_fd_);
            serial_fd_ = -1;
        }
    }

private:
    // =========================================================================
    //  Serial Port 操作 (POSIX termios)
    // =========================================================================

    /**
     * @brief 開啟 Serial Port (阻塞模式，供 reader thread 使用 poll + read)
     */
    bool open_serial(const std::string &port, int baudrate)
    {
        // O_RDWR: 讀寫模式
        // O_NOCTTY: 不要成為控制終端
        // 注意：不使用 O_NONBLOCK，reader thread 使用 poll() 做超時控制
        serial_fd_ = ::open(port.c_str(), O_RDWR | O_NOCTTY);
        if (serial_fd_ < 0) {
            RCLCPP_ERROR(this->get_logger(), "open() 失敗: %s (%s)",
                         port.c_str(), strerror(errno));
            return false;
        }

        struct termios tty{};
        if (tcgetattr(serial_fd_, &tty) != 0) {
            RCLCPP_ERROR(this->get_logger(), "tcgetattr() 失敗: %s", strerror(errno));
            ::close(serial_fd_);
            serial_fd_ = -1;
            return false;
        }

        // --- 鮑率 ---
        speed_t baud_flag = baud_to_flag(baudrate);
        cfsetispeed(&tty, baud_flag);
        cfsetospeed(&tty, baud_flag);

        // --- 8N1, 無流控 (Jetson 側是 8N1, Arduino 側 RS485 才是 8E1) ---
        tty.c_cflag &= ~PARENB;         // 無同位元
        tty.c_cflag &= ~CSTOPB;         // 1 stop bit
        tty.c_cflag &= ~CSIZE;
        tty.c_cflag |= CS8;             // 8 data bits
        tty.c_cflag &= ~CRTSCTS;        // 無硬體流控
        tty.c_cflag |= CREAD | CLOCAL;  // 啟用接收，忽略 modem 控制

        // --- Raw mode ---
        tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
        tty.c_iflag &= ~(IXON | IXOFF | IXANY);
        tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);
        tty.c_oflag &= ~OPOST;

        // --- 阻塞式讀取 (VMIN=1, VTIME=1) ---
        // VMIN=1: 至少收到 1 Byte 才回傳
        // VTIME=1: 但最多等 100ms (0.1s) 就回傳，方便 thread 檢查 running_ flag
        tty.c_cc[VMIN]  = 1;
        tty.c_cc[VTIME] = 1;

        if (tcsetattr(serial_fd_, TCSANOW, &tty) != 0) {
            RCLCPP_ERROR(this->get_logger(), "tcsetattr() 失敗: %s", strerror(errno));
            ::close(serial_fd_);
            serial_fd_ = -1;
            return false;
        }

        tcflush(serial_fd_, TCIOFLUSH);
        return true;
    }

    /**
     * @brief 將整數鮑率轉成 termios speed_t flag
     */
    static speed_t baud_to_flag(int baudrate)
    {
        switch (baudrate) {
            case 9600:   return B9600;
            case 19200:  return B19200;
            case 38400:  return B38400;
            case 57600:  return B57600;
            case 115200: return B115200;
            case 230400: return B230400;
            case 460800: return B460800;
            case 921600: return B921600;
            default:     return B115200;
        }
    }

    /**
     * @brief Thread-safe 寫入 Serial Port
     */
    void write_serial(const std::string &data)
    {
        std::lock_guard<std::mutex> lock(write_mutex_);
        if (serial_fd_ < 0) return;

        ssize_t total = 0;
        ssize_t len = static_cast<ssize_t>(data.size());
        while (total < len) {
            ssize_t n = ::write(serial_fd_, data.c_str() + total, len - total);
            if (n < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    usleep(100);
                    continue;
                }
                RCLCPP_ERROR(this->get_logger(), "write() 失敗: %s", strerror(errno));
                return;
            }
            total += n;
        }
        tcdrain(serial_fd_);
    }

    // =========================================================================
    //  獨立 Serial Reader Thread
    // =========================================================================

    /**
     * @brief 在獨立 thread 中執行，持續讀取 Serial Port
     *
     * 使用 poll() 做超時控制：
     *   - 有數據時立刻讀取，達到最低延遲
     *   - 無數據時每 100ms 醒來一次，檢查 running_ flag
     *   - 讀到 '\n' 就觸發 dispatch_json_line()
     */
    void reader_thread_func()
    {
        RCLCPP_INFO(this->get_logger(), "[Reader Thread] 已啟動");

        std::string line_buffer;
        line_buffer.reserve(256);

        struct pollfd pfd;
        pfd.fd = serial_fd_;
        pfd.events = POLLIN;

        while (running_.load()) {
            // poll() 最多等 100ms
            int ret = poll(&pfd, 1, 100);
            if (ret < 0) {
                if (errno == EINTR) continue;
                RCLCPP_ERROR(this->get_logger(), "[Reader Thread] poll() 錯誤: %s",
                             strerror(errno));
                break;
            }
            if (ret == 0) continue;  // 超時，重新檢查 running_

            // 有數據可讀
            char buf[512];
            ssize_t n = ::read(serial_fd_, buf, sizeof(buf) - 1);
            if (n <= 0) {
                if (n == 0) continue;  // EOF
                if (errno == EAGAIN || errno == EINTR) continue;
                RCLCPP_ERROR(this->get_logger(), "[Reader Thread] read() 錯誤: %s",
                             strerror(errno));
                break;
            }

            // 逐字元累積，遇到 '\n' 就分派
            for (ssize_t i = 0; i < n; ++i) {
                char c = buf[i];
                if (c == '\n') {
                    // 去除可能的 '\r'
                    if (!line_buffer.empty() && line_buffer.back() == '\r') {
                        line_buffer.pop_back();
                    }
                    if (!line_buffer.empty()) {
                        dispatch_json_line(line_buffer);
                    }
                    line_buffer.clear();
                } else {
                    line_buffer.push_back(c);
                    // 防止記憶體爆炸
                    if (line_buffer.size() > 1024) {
                        RCLCPP_WARN(this->get_logger(),
                                    "[Reader Thread] 行緩衝超過 1KB，清除");
                        line_buffer.clear();
                    }
                }
            }
        }

        RCLCPP_INFO(this->get_logger(), "[Reader Thread] 已結束");
    }

    /**
     * @brief 根據 JSON 中的 Key 值，分發到對應的 ROS Topic
     *
     * 分流規則：
     *   - 包含 "p1" → /raw_encoder_json      (里程計 JSON)
     *   - 包含 "pow" → /battery_state (電池電量)
     *   - 包含 "can_v" → /charge_status (充電站狀態)
     *   - 其他 → /serial_rx           (系統訊息等)
     *
     * 注意：這裡只做「字串子串匹配」，不做完整 JSON 解析。
     * 因為本節點是透明橋接器，完整解析由下游 kinematics_node 負責。
     * 使用 find() 而非 JSON parser 的原因是極致低延遲（避免額外 CPU 負擔）。
     */
    void dispatch_json_line(const std::string &line)
    {
        auto msg = std_msgs::msg::String();
        msg.data = line;

        if (line.find("\"p1\"") != std::string::npos) {
            // 里程計封包: {"p1":xxx,"p2":xxx}
            pub_raw_odom_->publish(msg);
        }
        else if (line.find("\"pow\"") != std::string::npos) {
            // 電池電量封包: {"pow":24.5}
            pub_battery_state_->publish(msg);
        }
        else if (line.find("\"can_v\"") != std::string::npos) {
            // 充電站狀態封包: {"can_v":0.00,"can_w":0.00,"c_st":0}
            pub_charge_status_->publish(msg);
        }
        else {
            // 系統訊息 / 未分類: {"sys":"AMR_READY..."} etc.
            pub_serial_rx_->publish(msg);
            RCLCPP_INFO(this->get_logger(), "Arduino 系統訊息: %s", line.c_str());
        }
    }

    // =========================================================================
    //  ROS Subscriber Callbacks
    // =========================================================================

    /**
     * @brief 收到 /motor_cmd → 寫入 Arduino
     *        訊息內容是完整 JSON 字串，本節點自動加上 '\n'
     */
    void motor_cmd_callback(const std_msgs::msg::String::SharedPtr msg)
    {
        std::string data = msg->data;
        if (data.empty()) return;
        if (data.back() != '\n') data.push_back('\n');
        write_serial(data);
        RCLCPP_DEBUG(this->get_logger(), "TX motor_cmd → %s", msg->data.c_str());
    }

    /**
     * @brief 收到 /charge_cmd → 寫入 Arduino
     *        預期格式: {"charge":1} 或 {"charge":0}
     */
    void charge_cmd_callback(const std_msgs::msg::String::SharedPtr msg)
    {
        std::string data = msg->data;
        if (data.empty()) return;
        if (data.back() != '\n') data.push_back('\n');
        write_serial(data);
        RCLCPP_INFO(this->get_logger(), "TX charge_cmd → %s", msg->data.c_str());
    }

    // =========================================================================
    //  成員變數
    // =========================================================================
    int serial_fd_;
    std::mutex write_mutex_;        // 僅保護 write，read 由獨立 thread 獨佔
    std::atomic<bool> running_;
    std::thread reader_thread_;

    // Publishers (分流輸出)
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_raw_odom_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_battery_state_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_charge_status_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_serial_rx_;

    // Subscribers (接收指令)
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_motor_cmd_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_charge_cmd_;
};

// =============================================================================
//  main
// =============================================================================
int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<SerialBridgeNode>();
    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}

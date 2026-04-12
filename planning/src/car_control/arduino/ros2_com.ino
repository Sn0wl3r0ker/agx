// ============================================================
//  差速履帶 AMR — Arduino Mega 最終生產版控制韌體 (整合充電與電量)
//  通訊格式：115200 bps, 8E1
//  核心防護：8ms 匯流排防撞、相位偏移排程、看門狗自動刷新
// ============================================================

#include <ArduinoJson.h>
#include <mcp_can.h>
#include <SPI.h>

#define ROS_SERIAL      Serial    
#define MODBUS_SERIAL   Serial2   

#define LEFT_SLAVE_ID   1
#define RIGHT_SLAVE_ID  2

// --- 充電站 CAN 配置 ---
const int SPI_CS_PIN = 10;
MCP_CAN CAN(SPI_CS_PIN);
bool if_exe_charge = false;

// --- 時序常數 ---
#define ENCODER_PUSH_MS 50   // Odom 推播頻率 (20Hz)
#define BATTERY_PUSH_MS 1000 // 電量推播頻率 (1Hz)
#define CAN_PUSH_MS     100  // CAN 充電站推播 (10Hz)
#define WATCHDOG_TIMEOUT_MS 500 

// --- 狀態變數 (相位偏移排程) ---
static unsigned long last_encoder_ms = 0;
static unsigned long last_battery_ms = 0;
static unsigned long last_can_ms     = 0;
static unsigned long last_cmd_ms     = 0;

static long last_valid_p1 = 0; 
static long last_valid_p2 = 0;

// --- JSON 接收緩衝區 ---
#define ROS_BUF_SIZE 128
static char ros_buf[ROS_BUF_SIZE];
static uint8_t ros_buf_idx = 0;

// --- 前向宣告 ---
uint16_t modbus_crc16(const uint8_t *data, uint16_t len);
bool modbus_read_position(uint8_t slave_id, long *out_position);
bool modbus_read_power(uint8_t slave_id, float *out_power);
void driver_set_speed(uint8_t slave_id, int16_t speed);

void setup() {
    ROS_SERIAL.begin(115200);
    MODBUS_SERIAL.begin(115200, SERIAL_8E1);
    
    delay(1000);

    // 初始化 CAN Bus
    if (CAN.begin(MCP_ANY, CAN_1000KBPS, MCP_8MHZ) == CAN_OK) {
        CAN.setMode(MCP_NORMAL);
        ROS_SERIAL.println("{\"sys\":\"CAN_INIT_OK\"}");
    } else {
        ROS_SERIAL.println("{\"warn\":\"CAN_INIT_FAIL\"}");
    }

    unsigned long now = millis();
    // 嚴格相位偏移，錯開 UART 發送瞬間，防止緩衝區阻塞
    last_encoder_ms = now;        // 0ms 基準
    last_can_ms     = now + 15;   // 晚 15ms 觸發
    last_battery_ms = now + 30;   // 晚 30ms 觸發
    last_cmd_ms     = now;

    ROS_SERIAL.println("{\"sys\":\"AMR_READY_115200_8E1\"}");
}

void loop() {
    ros_serial_read();

    unsigned long now = millis();

    if (now - last_encoder_ms >= ENCODER_PUSH_MS) {
        last_encoder_ms = now;
        push_encoder();
    }

    if (now - last_battery_ms >= BATTERY_PUSH_MS) {
        last_battery_ms = now;
        push_battery();
    }

    if (now - last_can_ms >= CAN_PUSH_MS) {
        last_can_ms = now;
        push_can_data();
    }

    watchdog_check();
}

// ============================================================
//  ROS 2 邏輯區塊
// ============================================================

void ros_serial_read() {
    while (ROS_SERIAL.available() > 0) {
        char c = (char)ROS_SERIAL.read();
        if (c == '\n') {
            ros_buf[ros_buf_idx] = '\0';
            if (ros_buf_idx > 0) parse_ros_command(ros_buf);
            ros_buf_idx = 0;
        } else if (c != '\r') {
            if (ros_buf_idx < ROS_BUF_SIZE - 1) ros_buf[ros_buf_idx++] = c;
            else ros_buf_idx = 0; 
        }
    }
}

void parse_ros_command(const char *json_str) {
    StaticJsonDocument<128> doc;
    DeserializationError err = deserializeJson(doc, json_str);
    if (err) return; 

    // 處理手動速度指令
    if (doc.containsKey("ls") && doc.containsKey("rs") && !if_exe_charge) {
        int16_t ls = (int16_t)doc["ls"].as<int>();
        int16_t rs = (int16_t)doc["rs"].as<int>();

        driver_set_speed(LEFT_SLAVE_ID, ls);
        delay(8); // ⚠️ 物理防撞
        driver_set_speed(RIGHT_SLAVE_ID, rs);
        last_cmd_ms = millis();
    }

    // 處理充電站開關指令 {"charge": 1} 或 {"charge": 0}
    if (doc.containsKey("charge")) {
        int charge_cmd = doc["charge"].as<int>();
        if (charge_cmd == 1) {
            if_exe_charge = true;
        } else {
            if_exe_charge = false;
            // 關閉時強制煞車並交回控制權
            driver_set_speed(LEFT_SLAVE_ID, 0);
            delay(8);
            driver_set_speed(RIGHT_SLAVE_ID, 0);
            last_cmd_ms = millis();
        }
    }
}

void watchdog_check() {
    unsigned long now = millis();
    if ((now - last_cmd_ms) > WATCHDOG_TIMEOUT_MS) {
        driver_set_speed(LEFT_SLAVE_ID, 0);
        delay(8);
        driver_set_speed(RIGHT_SLAVE_ID, 0);
        last_cmd_ms = now;
    }
}

void push_encoder() {
    long p1, p2;
    if (modbus_read_position(LEFT_SLAVE_ID, &p1)) last_valid_p1 = p1;
    delay(8); 
    if (modbus_read_position(RIGHT_SLAVE_ID, &p2)) last_valid_p2 = p2;

    ROS_SERIAL.print("{\"p1\":"); ROS_SERIAL.print(last_valid_p1);
    ROS_SERIAL.print(",\"p2\":"); ROS_SERIAL.print(last_valid_p2);
    ROS_SERIAL.println("}");
}

void push_battery() {
    float voltage = 0.0;
    // 透過 Modbus 讀取 0x0038 暫存器
    if (modbus_read_power(LEFT_SLAVE_ID, &voltage)) {
        ROS_SERIAL.print("{\"pow\":");
        ROS_SERIAL.print(voltage, 1);
        ROS_SERIAL.println("}");
    }
}

void push_can_data() {
    long unsigned int rxId;
    unsigned char len = 0;
    unsigned char data[8];

    if (CAN_MSGAVAIL == CAN.checkReceive()) {
        CAN.readMsgBuf(&rxId, &len, data);

        if (rxId == 0x182) {
            int16_t linear_raw = (data[0] << 8) | data[1];
            float linear_speed = linear_raw * 0.001;

            int16_t angular_raw = (data[4] << 8) | data[5];
            float angular_speed = angular_raw * 0.001;

            byte flag = data[6];
            bool is_charging = flag & (1 << 0);
            int c_st = is_charging ? 1 : 0;

            // 1. 推播充電狀態給 ROS 2
            ROS_SERIAL.print("{\"can_v\":"); ROS_SERIAL.print(linear_speed, 2);
            ROS_SERIAL.print(",\"can_w\":"); ROS_SERIAL.print(angular_speed, 2);
            ROS_SERIAL.print(",\"c_st\":"); ROS_SERIAL.print(c_st);
            ROS_SERIAL.println("}");

            // 2. 自動進站覆寫邏輯 (核心防護：帶有延遲與看門狗刷新)
            if (if_exe_charge) {
                if (is_charging) {
                    driver_set_speed(LEFT_SLAVE_ID, 0);
                    delay(8);
                    driver_set_speed(RIGHT_SLAVE_ID, 0);
                    if_exe_charge = false; // 已充到電，解除覆寫
                    last_cmd_ms = millis();
                } else {
                    int left_speed = -50;
                    int right_speed = -50;
                    byte speed = 100;
                    
                    if (linear_speed != 0) {
                        left_speed = -speed;
                        right_speed = -speed;
                    }
                    if (angular_speed < 0) {
                        right_speed -= speed;
                    } else if (angular_speed > 0) {
                        left_speed -= speed;
                    }

                    driver_set_speed(LEFT_SLAVE_ID, left_speed);
                    delay(8); // ⚠️ 物理防撞
                    driver_set_speed(RIGHT_SLAVE_ID, right_speed);
                    
                    last_cmd_ms = millis(); // ⚠️ 極度關鍵：告訴看門狗「我還活著，不要煞車」
                }
            }
        }
    }
}

// ============================================================
//  底層 Modbus 通訊區塊
// ============================================================

uint16_t modbus_crc16(const uint8_t *data, uint16_t len) {
    uint16_t crc = 0xFFFF;
    for (uint16_t i = 0; i < len; i++) {
        crc ^= (uint16_t)data[i];
        for (uint8_t j = 0; j < 8; j++) {
            if (crc & 0x0001) crc = (crc >> 1) ^ 0xA001;
            else crc >>= 1;
        }
    }
    return crc;
}

bool modbus_read_position(uint8_t slave_id, long *out_position) {
    uint8_t req[8] = {slave_id, 0x03, 0x00, 0x24, 0x00, 0x02, 0x00, 0x00};
    uint16_t crc = modbus_crc16(req, 6);
    req[6] = crc & 0xFF; req[7] = (crc >> 8) & 0xFF;

    while (MODBUS_SERIAL.available()) MODBUS_SERIAL.read();
    MODBUS_SERIAL.write(req, 8);
    MODBUS_SERIAL.flush();

    uint8_t resp[9];
    uint8_t idx = 0;
    unsigned long t0 = millis();

    while (idx < 9 && (millis() - t0) < 20) {
        if (MODBUS_SERIAL.available()) resp[idx++] = MODBUS_SERIAL.read();
    }

    if (idx < 9) return false;
    uint16_t recv_crc = (uint16_t)resp[7] | ((uint16_t)resp[8] << 8);
    if (recv_crc != modbus_crc16(resp, 7)) return false;
    if (resp[0] != slave_id || resp[1] != 0x03) return false;

    uint8_t raw[4] = {resp[6], resp[5], resp[4], resp[3]};
    memcpy(out_position, raw, 4);
    return true;
}

bool modbus_read_power(uint8_t slave_id, float *out_power) {
    // 讀取電壓：暫存器 0x0038，長度 1 Word
    uint8_t req[8] = {slave_id, 0x03, 0x00, 0x38, 0x00, 0x01, 0x00, 0x00};
    uint16_t crc = modbus_crc16(req, 6);
    req[6] = crc & 0xFF; req[7] = (crc >> 8) & 0xFF;

    while (MODBUS_SERIAL.available()) MODBUS_SERIAL.read();
    MODBUS_SERIAL.write(req, 8);
    MODBUS_SERIAL.flush();

    uint8_t resp[7]; // 回應長度為 7 Bytes
    uint8_t idx = 0;
    unsigned long t0 = millis();

    while (idx < 7 && (millis() - t0) < 20) {
        if (MODBUS_SERIAL.available()) resp[idx++] = MODBUS_SERIAL.read();
    }

    if (idx < 7) return false;
    uint16_t recv_crc = (uint16_t)resp[5] | ((uint16_t)resp[6] << 8);
    if (recv_crc != modbus_crc16(resp, 5)) return false;
    if (resp[0] != slave_id || resp[1] != 0x03) return false;

    uint16_t raw_val = (resp[3] << 8) | resp[4];
    *out_power = raw_val / 10.0;
    return true;
}

void driver_set_speed(uint8_t slave_id, int16_t speed) {
    uint8_t req[8] = {slave_id, 0x06, 0x00, 0x43, (uint8_t)(speed >> 8), (uint8_t)(speed & 0xFF), 0x00, 0x00};
    uint16_t crc = modbus_crc16(req, 6);
    req[6] = crc & 0xFF; req[7] = (crc >> 8) & 0xFF;

    while (MODBUS_SERIAL.available()) MODBUS_SERIAL.read();
    MODBUS_SERIAL.write(req, 8);
    MODBUS_SERIAL.flush();

    uint8_t discarded = 0;
    unsigned long t0 = millis();
    while (discarded < 8 && (millis() - t0) < 10) {
        if (MODBUS_SERIAL.available()) {
            MODBUS_SERIAL.read();
            discarded++;
        }
    }
}
import importlib

def get_msg_class(type_str):
    """
    動態取得 ROS Message Class
    範例: "std_msgs.msg.String" -> 載入 std_msgs.msg 模組，並回傳 String 類別
    """
    parts = type_str.split('.')
    module_name = '.'.join(parts[:-1])
    class_name = parts[-1]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def msg_to_dict(msg):
    """
    萬用轉換：將任意 ROS (1 或 2) 訊息物件遞迴轉換成 Python Dict
    以利 JSON 序列化傳輸。
    """
    if hasattr(msg, 'get_fields_and_field_types'): # ROS 2 的訊息
        fields = msg.get_fields_and_field_types().keys()
    elif hasattr(msg, '__slots__'): # ROS 1 的訊息
        fields = msg.__slots__
    else:
        return msg

    res = {}
    for f in fields:
        val = getattr(msg, f)
        if hasattr(val, '__slots__') or hasattr(val, 'get_fields_and_field_types'):
            res[f] = msg_to_dict(val)
        elif isinstance(val, list) or isinstance(val, tuple):
            res[f] = [msg_to_dict(v) if (hasattr(v, '__slots__') or hasattr(v, 'get_fields_and_field_types')) else v for v in val]
        else:
            res[f] = val
    return res

def dict_to_msg(d, msg):
    """
    萬用轉換：將 Python Dict 依照階層遞迴塞回給 ROS 訊息物件中
    傳入的 msg 必須是已經經過實例化的 ROS 物件 ()
    """
    for key, val in d.items():
        if not hasattr(msg, key):
            continue
        
        attr = getattr(msg, key)
        if isinstance(val, dict):
            dict_to_msg(val, attr)
        elif isinstance(val, list):
            # 目前簡單針對一維基礎型態 List（像 array）直接覆蓋
            # 若為 complex ros array，需要有類型推導來實例化 element
            setattr(msg, key, val)
        else:
            setattr(msg, key, val)
    return msg

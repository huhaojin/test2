import pandas as pd

# Data from the transcribed image
data = [
    ["(6)", "Validity of the model", "型号有效性", "To be mentioned"],
    ["(7)", "How long the proposed model will remain in production", "该型号预计还能维持生产多久 (生命周期)", "To be mentioned"],
    ["(8)", "Duration of after sales svc support", "售后服务支持时长", "To be mentioned"],
    ["(9)", "Number of sensors", "传感器数量", "To be mentioned"],
    ["(10)", "Low light capability.", "低光照/弱光性能", "To be mentioned"],
    ["(11)", "Surveillance camera must provide Full HD video output at min 30 fps.", "监控摄像机必须提供至少 30 fps 的全高清视频输出", "To be mentioned"],
    ["(12)", "Day and night capability at 0 lux.", "0 Lux 下的日夜视能力", "To be mentioned"],
    ["(13)", "Pan /Tilt /Zoom: 20-400 mm, continuous zoom, auto white balance or better.", "云台/俯仰/变焦: 20-400 mm, 连续变焦, 自动白平衡或更好", "To be mentioned"],
    ["(14)", "Ground resolution: 0.1 m (@FOV=0.6deg, Altitude: 10000 m) or better.", "地面分辨率: 0.1 m (@视场角=0.6度, 高度: 10000 m) 或更好", "To be mentioned"],
    ["(15)", "Picture element resolution: 0.25 m (Altitude: 10000 m, visibility: 15 km) or better.", "像元分辨率: 0.25 m (高度: 10000 m, 能见度: 15 km) 或更好", "To be mentioned"],
    ["(16)", "Maximum speed-height ration: ≥ 50°/s.", "最大速高比 (speed-height ratio): ≥ 50°/s", "To be mentioned"],
    ["(17)", "Single photography area: 1 km x 1 km (@FOV=14deg, Altitude:10000 m) or better.", "单次拍摄覆盖区域: 1 km x 1 km (@视场角=14度, 高度:10000 m) 或更好", "To be mentioned"],
    ["(18)", "Image sensor Type/name", "图像传感器类型/名称", "To be mentioned"],
    ["(19)", "Max image size and output format", "最大图像尺寸及输出格式", "To be mentioned"],
    ["(20)", "Power supply", "电源供应", "To be mentioned"],
    ["(21)", "Power consumption", "功耗", "To be mentioned"],
    ["(22)", "Temperature", "工作温度", "To be mentioned"],
    ["(23)", "Weight", "重量", "To be mentioned"],
    ["(24)", "Width", "宽度", "To be mentioned"],
    ["(25)", "Height", "高度", "To be mentioned"],
    ["(26)", "Weather sealed", "气候密封性 (防水防尘)", "To be mentioned"],
    ["(27)", "Anti-vibration damping mount", "防抖减震底座", "To be mentioned"],
    ["(28)", "Lense size", "镜头尺寸", "To be mentioned"],
    ["(29)", "Maintainace cost", "维护成本", "To be provided"],
    ["(30)", "Onboard recording", "机载录像/板载存储", "To be mentioned"],
    ["(31)", "Any other option", "其他选项/功能", "To be mentioned"]
]

# Create DataFrame
columns = ["Item No.", "Requirement / Description (English)", "Requirement / Description (Chinese)", "Status Required"]
df = pd.DataFrame(data, columns=columns)

# Add an empty column for user input
df["Supplier Response / Remarks"] = ""

# Save to Excel
file_path = "Technical_Specifications_CN_EN.xlsx"
df.to_excel(file_path, index=False)

file_path
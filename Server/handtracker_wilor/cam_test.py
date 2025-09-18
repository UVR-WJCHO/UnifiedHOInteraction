import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()
if len(devices) == 0:
    print("❌ No RealSense devices connected.")
else:
    for dev in devices:
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        print(f"✅ Found device: {name} (Serial: {serial})")

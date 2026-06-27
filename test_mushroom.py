"""测试蘑菇数量识别功能"""
import base64
import json
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from app.engines import get_recognizer


def test_mushroom(image_path: str, app: str = "poler"):
    """测试指定图片的蘑菇数量识别"""
    print(f"\n{'='*60}")
    print(f"测试图片: {image_path}")
    print(f"目标 APP: {app}")
    print(f"{'='*60}")
    
    # 加载图片并转 base64
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        print(f"图片加载成功，大小: {len(image_bytes)} bytes")
    except Exception as e:
        print(f"图片加载失败: {e}")
        return
    
    # 获取识别器并识别
    try:
        recognizer = get_recognizer(app)
        print(f"识别器初始化成功")
        
        print("\n开始识别...")
        result = recognizer.recognize(image_base64, parse_all=True)
        print(f"识别完成")
    except Exception as e:
        print(f"识别失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 打印完整结果（格式化）
    print("\n识别结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 重点展示蘑菇数量
    mushroom_count = result["table_info"].get("mushroom_count")
    print(f"\n{'='*60}")
    print(f"蘑菇数量识别结果: {mushroom_count}")
    print(f"预期结果: 90")
    print(f"识别{'成功 ✓' if mushroom_count == 90 else '失败 ✗'}")
    print(f"{'='*60}")
    
    return result


if __name__ == "__main__":
    # 默认测试 16.jpg
    test_image = "/Users/bytedance/go/src/kingjxu/dezhou/16.jpg"
    
    # 如果命令行传入了图片路径，则使用传入的路径
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    
    # 检测图片是否存在
    if not Path(test_image).exists():
        print(f"错误: 图片文件不存在 - {test_image}")
        sys.exit(1)
    
    test_mushroom(test_image, app="poler")
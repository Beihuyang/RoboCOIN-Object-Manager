"""Offline Chinese display names for SAM3 semantic discovery prompts.

Resolution order for ``prompt_zh``: built-in ``NOUN_ZH`` dictionary first, then
the generated ``noun_translations_cache.json`` (produced offline by
``precompute_noun_translations.py`` using a lightweight translation API), then
modifier + base composition against both dictionaries.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


NOUN_ZH = {
    "object": "通用物体", "box": "盒子", "block": "积木块", "trash": "垃圾",
    "test tube": "试管", "cardboard": "纸板", "storage box": "收纳盒",
    "tableware": "餐具", "grape": "葡萄", "building block": "积木",
    "cube": "立方体", "clothes": "衣物", "food": "食物", "cake": "蛋糕",
    "bowl": "碗", "cable": "线缆", "cup": "杯子", "tube": "管子",
    "plate": "盘子", "fruit": "水果", "drawer": "抽屉", "bread": "面包",
    "bottle": "瓶子", "flower": "花", "basket": "篮子", "egg": "鸡蛋",
    "apple": "苹果", "book": "书", "garbage": "垃圾", "vegetable": "蔬菜",
    "baozi": "包子", "shelf": "架子", "square": "方形物体",
    "coffee bean": "咖啡豆", "tissue": "纸巾", "towel": "毛巾",
    "pillow": "枕头", "toy": "玩具", "pen": "笔", "chopstick": "筷子",
    "shoes": "鞋", "shoe": "鞋", "sponge": "海绵", "ice cream": "冰淇淋",
    "orange": "橙子", "tissue paper": "纸巾", "bag": "袋子", "banana": "香蕉",
    "drink": "饮料", "toiletry": "洗漱用品", "button": "按钮", "cap": "盖子",
    "cleaner": "清洁用品", "medicine": "药品", "tray": "托盘",
    "paper box": "纸盒", "ball": "球", "door": "门", "mouse": "鼠标",
    "lemon": "柠檬", "lid": "盖子", "page": "纸页", "paper": "纸",
    "potato": "土豆", "shrimp": "虾", "zipper": "拉链", "cream": "乳霜",
    "utensil": "器具", "beverage": "饮料", "breakfast": "早餐",
    "coffee": "咖啡", "mango": "芒果", "pear": "梨", "sink": "水槽",
    "slipper": "拖鞋", "tape": "胶带", "trash can": "垃圾桶",
    "water bottle": "水瓶", "bowls": "碗", "peach": "桃子", "tool": "工具",
    "beaker": "烧杯", "electronics": "电子设备", "meal": "餐食",
    "opening": "开口", "pants": "裤子", "pepper": "辣椒", "phone": "手机",
    "pomegranate": "石榴", "sharpener": "削笔器", "shorts": "短裤",
    "takeout box": "外卖盒", "bell pepper": "甜椒", "card": "卡片",
    "pencil sharpener": "卷笔刀", "pumpkin": "南瓜", "rubik": "魔方",
    "tissue box": "纸巾盒", "yellow box": "黄色盒子", "battery": "电池",
    "beer": "啤酒", "board": "板子", "cake plate": "蛋糕盘", "car": "汽车",
    "chair": "椅子", "coffee cup": "咖啡杯", "guitar": "吉他",
    "holder": "支架", "hole": "孔", "jean": "牛仔裤", "kettle": "水壶",
    "label": "标签", "lamp": "灯", "magnet": "磁铁", "marker": "记号笔",
    "microwave oven": "微波炉", "paper cup": "纸杯", "pliers": "钳子",
    "pot": "锅", "purple block": "紫色积木块", "ring": "环",
    "router": "路由器", "shirt": "衬衫", "spoon": "勺子", "sweep": "清扫工具",
    "table tennis ball": "乒乓球", "toilet": "马桶", "toy car": "玩具汽车",
    "white bag": "白色袋子", "black marker": "黑色记号笔",
    "blue plate": "蓝色盘子", "brown bag": "棕色袋子",
    "brown basket": "棕色篮子", "brown plate": "棕色盘子", "can": "罐子",
    "dish": "碟子", "doll": "玩偶", "equipment": "设备", "khaki": "卡其色物体",
    "laptop": "笔记本电脑", "microwave": "微波炉", "onion": "洋葱",
    "oven": "烤箱", "pan": "平底锅", "peeler": "削皮器", "pencil": "铅笔",
    "pineapple": "菠萝", "pink bowl": "粉色碗", "pyramid": "棱锥体",
    "racket": "球拍", "reset": "复位按钮", "shark": "鲨鱼", "switch": "开关",
    "tomato": "番茄", "triangle": "三角形物体", "vase": "花瓶",
    "white box": "白色盒子", "wipe": "湿巾", "xylophone": "木琴",
    "yellow basket": "黄色篮子", "yellow block": "黄色积木块",
    "accessory": "配件", "air": "空气", "angle iron": "角铁", "bank": "储存装置",
    "bar": "条状物", "bars": "条状物", "basin": "盆", "bbs": "塑料弹珠",
    "bean": "豆子", "bed": "床", "belly": "腹部", "bin": "收纳箱",
    "black tablecloth": "黑色桌布", "blackboard": "黑板", "blender": "搅拌器",
    "blue test tube": "蓝色试管", "blue tray": "蓝色托盘", "bluetooth": "蓝牙设备",
    "brain": "大脑模型", "brown bowl": "棕色碗", "bucket": "桶", "bulb": "灯泡",
    "burger": "汉堡", "cabinet": "柜子", "calculator": "计算器", "call": "电话设备",
    "canned food": "罐头食品", "capsule": "胶囊", "carton": "纸盒", "case": "盒套",
    "catch": "接取物", "central": "中央设备", "child": "儿童用品", "chop": "切割物",
    "click": "按键", "clip": "夹子", "clothes basket": "衣物篮", "computer": "电脑",
    "container": "容器", "cook": "炊具", "cookie": "饼干", "cookie cup": "饼干杯",
    "cosmetic": "化妆品", "counter": "操作台", "countertop": "台面", "cover": "盖子",
    "cupboard": "橱柜", "curtain": "窗帘", "cut": "切割件", "desk": "书桌",
    "dial": "旋钮", "diamond": "菱形物体", "document": "文件", "dog doll": "小狗玩偶",
    "dress shirt": "正装衬衫", "dryer": "吹风机", "duck": "鸭子玩具",
    "dumpling": "饺子", "egg yolk": "蛋黄", "eight": "八号物体", "empty": "空容器",
    "fake": "仿真物体", "five": "五号物体", "flat": "扁平物体", "four": "四号物体",
    "freezer": "冷冻柜", "fruit salad": "水果沙拉", "garden": "园艺用品",
    "glass": "玻璃杯", "glasses": "眼镜", "glasses case": "眼镜盒",
    "graphics": "图形物体", "gray plate": "灰色盘子", "green tablecloth": "绿色桌布",
    "grey tray": "灰色托盘", "hair dryer": "吹风机", "hamburger": "汉堡",
    "hanger": "衣架", "hose": "软管", "hourglass": "沙漏", "iron": "熨斗",
    "knife": "刀", "lab": "实验室用品", "landline": "座机", "machine": "机器",
    "marble": "弹珠", "milk": "牛奶", "mirror": "镜子", "mobile": "移动设备",
    "mobile phone": "手机", "mung bean": "绿豆", "needle": "针", "network": "网络设备",
    "nightstand": "床头柜", "nose": "尖嘴部件", "notebook": "笔记本",
    "obstacle": "障碍物", "parcel": "包裹", "pastry": "糕点", "penguin": "企鹅玩具",
    "piano": "钢琴", "pink cup": "粉色杯子", "pink tray": "粉色托盘", "pipe": "管道",
    "piston": "活塞", "plug": "插头", "plunger": "柱塞", "pork belly": "五花肉",
    "portable": "便携设备", "post": "柱子", "power bank": "充电宝", "rabbit": "兔子玩具",
    "red tablecloth": "红色桌布", "remote": "遥控器", "rice": "大米", "rod": "杆子",
    "ruler": "尺子", "salad": "沙拉", "sandbag": "沙袋", "sandwich": "三明治",
    "scallion": "葱", "screw": "螺丝", "seal": "封口件", "sensor card": "感应卡",
    "shoebox": "鞋盒", "short sleeve": "短袖衣物", "six": "六号物体", "sleeve": "衣袖",
    "socket": "插座", "soda": "汽水", "sponge cake": "海绵蛋糕",
    "standing": "立式物体", "stationery": "文具", "steamer": "蒸笼", "sticker": "贴纸",
    "straw": "吸管", "sweet potato": "红薯", "sword": "剑形玩具", "syringe": "注射器",
    "table tennis": "乒乓球用品", "tablecloth": "桌布", "takeout bag": "外卖袋",
    "tea": "茶", "teaset": "茶具", "tennis ball": "网球", "text": "文本载体",
    "thermometer": "温度计", "three": "三号物体", "tidy": "整理用品",
    "tiger": "老虎玩具", "tub": "盆桶", "tumbler": "随行杯", "umbrella": "雨伞",
    "upper": "上层物体", "wallpaper": "壁纸", "warehouse": "仓储用品",
    "wet wipe": "湿巾", "white basket": "白色篮子", "white tablecloth": "白色桌布",
    "yellow test tube": "黄色试管", "yellow tray": "黄色托盘", "yolk": "蛋黄",
    "zip": "拉链",
}

# Product, food, furniture, garment, tool and laboratory terms that occur in
# the current review manifests. Colour/size variants are composed below.
NOUN_ZH.update({
    "adhesive tape": "胶带", "air conditioner": "空调", "apparatus": "器材",
    "arch": "拱形件", "avocado": "牛油果", "bacon": "培根", "balcony": "阳台",
    "ballpoint pen": "圆珠笔", "bamboo": "竹制品", "bandage": "绷带",
    "barrel": "桶", "base": "底座", "bath towel": "浴巾", "bathroom": "浴室",
    "bear": "熊玩具", "beater": "搅拌器", "beer glass": "啤酒杯",
    "beer mug": "啤酒杯", "biscuit": "饼干", "blackboard eraser": "黑板擦",
    "blade": "刀片", "blanket": "毯子", "bookshelf": "书架", "bowel": "碗",
    "bracket": "支架", "bread maker": "面包机", "brick": "砖块", "broom": "扫帚",
    "brush": "刷子", "bull": "公牛玩具", "bullet": "子弹形物体", "bun": "面包卷",
    "buns": "面包卷", "cabbage": "卷心菜", "candle": "蜡烛", "candy": "糖果",
    "canister": "罐子", "canned cola": "罐装可乐", "cardboard box": "纸箱",
    "carpet": "地毯", "carrot": "胡萝卜", "cart": "推车", "ceramic": "陶瓷制品",
    "chalkboard": "黑板", "charger": "充电器", "cheese": "奶酪", "cherry": "樱桃",
    "chewing gum": "口香糖", "chili": "辣椒", "chili pepper": "辣椒",
    "chinese cabbage": "白菜", "chocolate": "巧克力", "chocolate cake": "巧克力蛋糕",
    "chopping board": "砧板", "circle": "圆形物体", "clamp": "夹具", "claw": "夹爪",
    "cleanser": "清洁剂", "cloth": "布", "clothes hanger": "衣架",
    "clothesline": "晾衣绳", "clothing": "衣物", "coil": "线圈", "coke": "可乐",
    "cola": "可乐", "column": "柱状物", "comb": "梳子", "compartment": "隔层",
    "compass": "圆规", "component": "零件", "controller": "控制器",
    "conveyor": "传送装置", "conveyor belt": "传送带", "cooker": "炊具",
    "cord": "线绳", "corn": "玉米", "cotton": "棉织物", "covering": "覆盖物",
    "cranberry juice": "蔓越莓汁", "croissant": "牛角包", "cucumber": "黄瓜",
    "cuff": "袖口", "cutlery": "餐具", "cutting board": "砧板", "cylinder": "圆柱体",
    "dagger": "匕首玩具", "denim": "牛仔布", "desk phone": "桌面电话",
    "detergent": "清洁剂", "device": "设备", "dish rack": "碗碟架",
    "dispenser": "分配器", "dolphin": "海豚玩具", "donut": "甜甜圈",
    "doorknob": "门把手", "drawers": "抽屉", "drawing board": "画板",
    "dumplings": "饺子", "dustpan": "簸箕", "earphone": "耳机", "eggplant": "茄子",
    "eggs": "鸡蛋", "electric kettle": "电水壶", "enclosure": "外壳",
    "eraser": "橡皮擦", "erlenmeyer flask": "锥形瓶", "ethernet cable": "网线",
    "eyeglass": "眼镜", "fabric": "布料", "faucet": "水龙头", "fiber": "纤维制品",
    "file": "文件夹", "foam": "泡沫材料", "foot": "脚部模型", "fork": "叉子",
    "frame": "框架", "french fries": "薯条", "fridge": "冰箱", "frying pan": "平底锅",
    "fudge": "软糖", "garment": "衣物", "glass cup": "玻璃杯", "glue": "胶水",
    "gourd": "葫芦", "graduated cylinder": "量筒", "grain": "谷物",
    "handbag": "手提包", "handle": "把手", "handset": "电话听筒", "hard disk": "硬盘",
    "hard drive": "硬盘", "hook": "挂钩", "hub": "集线器", "iced tea": "冰茶",
    "incense": "香薰", "index finger": "食指模型", "ingredient": "食材", "ink": "墨水",
    "juice": "果汁", "key": "钥匙", "keyboard": "键盘", "kiwi": "猕猴桃",
    "laundry detergent": "洗衣液", "leaf": "叶片", "leftovers": "剩余食物",
    "lego": "乐高积木", "lense": "镜片", "lettuce": "生菜", "lime": "青柠",
    "linen": "亚麻布", "locker": "储物柜", "long sleeve": "长袖衣物",
    "loofah": "沐浴球", "mangosteen": "山竹", "marbles": "弹珠",
    "marker pen": "记号笔", "mat": "垫子", "measuring cup": "量杯", "meat": "肉",
    "melon": "甜瓜", "mesh": "网状物", "metal": "金属物体", "mint candy": "薄荷糖",
    "mosquito": "蚊虫模型", "mouse mat": "鼠标垫", "mousepad": "鼠标垫", "mug": "马克杯",
    "net": "网", "noodle": "面条", "note": "便签", "nozzle": "喷嘴", "oolong": "乌龙茶",
    "oyster mushroom": "平菇", "pack": "包装", "packet": "小包装袋", "pad": "垫片",
    "pant leg": "裤腿", "papers": "纸张", "patty": "肉饼", "pellet": "颗粒物",
    "persimmon": "柿子", "picker": "夹取器", "pigment": "颜料", "pillar": "柱子",
    "pizza": "披萨", "plastic": "塑料物体", "plastic bag": "塑料袋",
    "plate rack": "盘架", "platform": "平台", "playing cards": "扑克牌",
    "plier": "钳子", "plugboard": "插线板", "plush": "毛绒玩具", "porcelain": "瓷器",
    "pork": "猪肉", "poster": "海报", "potato chips": "薯片", "pouch": "小袋",
    "prawn": "虾", "puff": "泡芙", "puppy": "小狗玩具", "quilt": "被子",
    "rack": "架子", "radish": "萝卜", "rag": "抹布", "reagent": "试剂",
    "receiver": "接收器", "refrigerator": "冰箱", "repellent": "驱虫剂",
    "residue": "残留物", "rings": "环", "roll": "卷状物", "rubber": "橡胶物体",
    "rubik cube": "魔方", "salt shaker": "盐罐", "sand": "沙子", "sauce": "酱料",
    "sausage": "香肠", "scale": "秤", "scanner": "扫描仪", "scissors": "剪刀",
    "scoop": "勺铲", "scratcher": "抓挠器", "sensor": "传感器", "shampoo": "洗发水",
    "sheet": "薄片", "shortbread": "酥饼", "shovel": "铲子", "shower": "花洒",
    "shuttlecock": "羽毛球", "sign": "标牌", "slab": "板块", "slice": "切片",
    "slot": "槽口", "snack": "零食", "soap": "肥皂", "soup": "汤",
    "soup spoon": "汤勺", "soy sauce": "酱油", "sphere": "球体", "spiral": "螺旋件",
    "spring": "弹簧", "square block": "方形积木", "stainless steel": "不锈钢物体",
    "stamp": "印章", "stand": "支架", "stapler": "订书机", "steel": "钢制物体",
    "stick": "棒状物", "stove": "炉灶", "strainer": "滤网", "strawberry": "草莓",
    "strings": "绳线", "strip": "条状物", "swab": "棉签", "sweatshirt": "卫衣",
    "swiss roll": "瑞士卷", "tabletop": "桌面", "tangerine": "橘子", "tap": "水龙头",
    "tart": "挞", "tea bag": "茶包", "tea leaf": "茶叶", "teacup": "茶杯",
    "teapot": "茶壶", "telephone": "电话", "tester": "测试仪", "thermos": "保温杯",
    "timer": "计时器", "tin": "铁罐", "toast": "吐司", "toaster": "烤面包机",
    "toilet bowl": "马桶", "toilet paper": "卫生纸", "tongs": "夹子",
    "toothbrush": "牙刷", "toothpaste": "牙膏", "toy dog": "小狗玩具",
    "tripod": "三脚架", "trouser": "裤子", "turntable": "转盘", "ukulele": "尤克里里",
    "vacuum": "吸尘器", "valve": "阀门", "vitamin b": "维生素B瓶",
    "waffle": "华夫饼", "waistband": "腰带", "wall": "墙面",
    "wallpaper knife": "壁纸刀", "wardrobe": "衣柜", "washbasin": "洗手盆",
    "washer": "洗衣机", "waste": "废弃物", "waste paper": "废纸",
    "wastebasket": "废纸篓", "wet tissue": "湿巾", "whale": "鲸鱼玩具",
    "whisk": "打蛋器", "window": "窗户", "wine": "葡萄酒", "wine bottle": "酒瓶",
    "wing": "翼状件", "wire": "电线", "workbench": "工作台", "yoghurt": "酸奶",
    "yogurt": "酸奶", "zucchini": "西葫芦",
})

MODIFIER_ZH = {
    "black": "黑色", "blue": "蓝色", "brown": "棕色", "gray": "灰色",
    "grey": "灰色", "green": "绿色", "orange": "橙色", "pink": "粉色",
    "purple": "紫色", "red": "红色", "white": "白色", "yellow": "黄色",
    "teal": "青绿色", "gold": "金色", "silver": "银色", "dark": "深色",
    "large": "大号", "small": "小号", "mini": "迷你", "upper": "上层",
    "lower": "下层", "flat": "扁平", "portable": "便携式", "standing": "立式",
    "upright": "直立式", "disposable": "一次性", "fake": "仿真",
}


def _default_cache_path() -> Path:
    """Path of the generated translation cache, overridable for tests/deploys."""
    override = os.environ.get("NOUN_TRANSLATIONS_CACHE", "").strip()
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent / "noun_translations_cache.json"


def _read_extra_cache() -> dict[str, str]:
    """Load the generated cache as a normalized english -> chinese mapping."""
    path = _default_cache_path()
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    extra = {}
    if isinstance(data, dict):
        for key, value in data.items():
            key = normalize_prompt(key)
            if key and isinstance(value, str) and value.strip():
                extra[key] = value.strip()
    return extra


def reload_extra_cache() -> dict[str, str]:
    """Re-read noun_translations_cache.json (e.g. after a refresh)."""
    global EXTRA_NOUN_ZH
    EXTRA_NOUN_ZH = _read_extra_cache()
    return EXTRA_NOUN_ZH


EXTRA_NOUN_ZH = {}


def normalize_prompt(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("_", " ").strip().lower())


EXTRA_NOUN_ZH = _read_extra_cache()


def prompt_zh(value: object) -> str:
    """Return a Chinese-only review label without changing the model prompt."""
    prompt = normalize_prompt(value)
    if not prompt:
        return "未命名物体"
    exact = NOUN_ZH.get(prompt)
    if exact is None:
        exact = EXTRA_NOUN_ZH.get(prompt)
    if exact:
        return exact
    words = prompt.split()
    modifiers = []
    while words and words[0] in MODIFIER_ZH:
        modifiers.append(MODIFIER_ZH[words.pop(0)])
    if modifiers and words:
        base = NOUN_ZH.get(" ".join(words))
        if base is None:
            base = EXTRA_NOUN_ZH.get(" ".join(words))
        if base:
            return "".join(modifiers) + base
    return "待核对物体"

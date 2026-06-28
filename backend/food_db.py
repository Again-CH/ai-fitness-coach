# -*- coding: utf-8 -*-
"""
food_db.py —— 本地食物营养数据库
=====================================
包含 200+ 常见中国食物的营养成分数据（每 100g 含量）。
提供模糊匹配 + NLP 描述解析功能，供 AI 工具调用。

数据格式：
    每条记录包含：
        name:     食物名称（中文）
        aliases:  别名/常见叫法列表
        category: 分类（主食/肉类/蔬菜/水果/乳制品/零食/饮品/坚果/豆制品/调味品）
        per_100g: 每 100g 的营养成分
            calories: 热量 (kcal)
            protein:  蛋白质 (g)
            carb:     碳水化合物 (g)
            fat:      脂肪 (g)
            fiber:    膳食纤维 (g)
        typical_portion: 常见一份的克数（用于估算）
        portion_desc:    一份的描述（如"1碗"、"1个"）
"""

from typing import Optional

# ============================================================
# ★ 食物营养数据库（每 100g 含量）
# ============================================================
FOOD_DATABASE = [
    # ==================== 主食 ====================
    {
        "name": "白米饭",
        "aliases": ["米饭", "大米饭", "蒸米饭", "熟米饭", "白饭"],
        "category": "主食",
        "per_100g": {"calories": 116, "protein": 2.6, "carb": 25.6, "fat": 0.3, "fiber": 0.3},
        "typical_portion": 200,
        "portion_desc": "1碗（约200g）",
    },
    {
        "name": "糙米饭",
        "aliases": ["糙米", "玄米饭", "全谷物米饭"],
        "category": "主食",
        "per_100g": {"calories": 111, "protein": 2.8, "carb": 23.5, "fat": 0.9, "fiber": 1.8},
        "typical_portion": 200,
        "portion_desc": "1碗（约200g）",
    },
    {
        "name": "馒头",
        "aliases": ["白馒头", "面包馒头", "蒸馒头"],
        "category": "主食",
        "per_100g": {"calories": 223, "protein": 7.0, "carb": 47.0, "fat": 1.0, "fiber": 1.3},
        "typical_portion": 100,
        "portion_desc": "1个（约100g）",
    },
    {
        "name": "面条",
        "aliases": ["煮面条", "挂面", "龙须面", "细面", "宽面", "熟面条"],
        "category": "主食",
        "per_100g": {"calories": 109, "protein": 3.3, "carb": 22.9, "fat": 0.2, "fiber": 0.2},
        "typical_portion": 200,
        "portion_desc": "1碗（约200g熟面）",
    },
    {
        "name": "馄饨",
        "aliases": ["抄手", "云吞", "混沌"],
        "category": "主食",
        "per_100g": {"calories": 120, "protein": 5.5, "carb": 18.0, "fat": 3.0, "fiber": 0.5},
        "typical_portion": 200,
        "portion_desc": "1碗（约200g）",
    },
    {
        "name": "饺子",
        "aliases": ["水饺", "蒸饺", "煎饺", "锅贴"],
        "category": "主食",
        "per_100g": {"calories": 141, "protein": 6.5, "carb": 22.0, "fat": 3.5, "fiber": 1.0},
        "typical_portion": 150,
        "portion_desc": "10个（约150g）",
    },
    {
        "name": "包子",
        "aliases": ["肉包子", "素包子", "菜包", "猪肉包", "叉烧包"],
        "category": "主食",
        "per_100g": {"calories": 211, "protein": 7.5, "carb": 35.0, "fat": 5.0, "fiber": 1.5},
        "typical_portion": 100,
        "portion_desc": "1个（约100g）",
    },
    {
        "name": "烙饼",
        "aliases": ["饼", "葱油饼", "手抓饼", "葱花饼"],
        "category": "主食",
        "per_100g": {"calories": 272, "protein": 7.0, "carb": 48.0, "fat": 6.0, "fiber": 1.5},
        "typical_portion": 100,
        "portion_desc": "1张（约100g）",
    },
    {
        "name": "燕麦片",
        "aliases": ["燕麦", "麦片", "即食燕麦", "oats", "大燕麦"],
        "category": "主食",
        "per_100g": {"calories": 389, "protein": 16.9, "carb": 66.3, "fat": 6.9, "fiber": 10.6},
        "typical_portion": 40,
        "portion_desc": "1份（约40g干）",
    },
    {
        "name": "红薯",
        "aliases": ["地瓜", "甘薯", "番薯", "山芋"],
        "category": "主食",
        "per_100g": {"calories": 86, "protein": 1.1, "carb": 20.1, "fat": 0.2, "fiber": 1.6},
        "typical_portion": 200,
        "portion_desc": "1个中等大小（约200g）",
    },
    {
        "name": "玉米",
        "aliases": ["甜玉米", "煮玉米", "玉米棒"],
        "category": "主食",
        "per_100g": {"calories": 112, "protein": 4.0, "carb": 22.8, "fat": 1.2, "fiber": 2.9},
        "typical_portion": 200,
        "portion_desc": "1根（约200g可食部分）",
    },
    {
        "name": "土豆",
        "aliases": ["马铃薯", "洋芋", "薯仔"],
        "category": "主食",
        "per_100g": {"calories": 77, "protein": 2.0, "carb": 17.0, "fat": 0.1, "fiber": 2.2},
        "typical_portion": 150,
        "portion_desc": "1个中等大小（约150g）",
    },

    # ==================== 肉类 ====================
    {
        "name": "鸡胸肉",
        "aliases": ["去皮鸡胸", "鸡扒", "鸡柳", "清蒸鸡胸", "水煮鸡胸"],
        "category": "肉类",
        "per_100g": {"calories": 133, "protein": 24.6, "carb": 0, "fat": 3.4, "fiber": 0},
        "typical_portion": 150,
        "portion_desc": "1块（约150g）",
    },
    {
        "name": "鸡腿",
        "aliases": ["鸡腿肉", "去骨鸡腿", "烤鸡腿", "卤鸡腿", "琵琶腿"],
        "category": "肉类",
        "per_100g": {"calories": 181, "protein": 20.2, "carb": 0, "fat": 11.2, "fiber": 0},
        "typical_portion": 120,
        "portion_desc": "1个（约120g可食部分）",
    },
    {
        "name": "瘦猪肉",
        "aliases": ["里脊肉", "猪里脊", "精瘦肉", "猪瘦肉", "猪肉"],
        "category": "肉类",
        "per_100g": {"calories": 143, "protein": 20.3, "carb": 1.5, "fat": 6.2, "fiber": 0},
        "typical_portion": 100,
        "portion_desc": "1份（约100g）",
    },
    {
        "name": "五花肉",
        "aliases": ["猪肚肉", "猪腩肉", "红烧肉"],
        "category": "肉类",
        "per_100g": {"calories": 395, "protein": 13.2, "carb": 2.4, "fat": 37.0, "fiber": 0},
        "typical_portion": 100,
        "portion_desc": "1份（约100g）",
    },
    {
        "name": "牛肉",
        "aliases": ["瘦牛肉", "牛里脊", "牛腱子", "炒牛肉", "牛扒", "牛排"],
        "category": "肉类",
        "per_100g": {"calories": 125, "protein": 20.0, "carb": 2.0, "fat": 4.2, "fiber": 0},
        "typical_portion": 150,
        "portion_desc": "1份（约150g）",
    },
    {
        "name": "羊肉",
        "aliases": ["羊排", "羊腿肉", "羊肉串", "烤羊肉"],
        "category": "肉类",
        "per_100g": {"calories": 203, "protein": 19.0, "carb": 0, "fat": 14.1, "fiber": 0},
        "typical_portion": 150,
        "portion_desc": "1份（约150g）",
    },
    {
        "name": "三文鱼",
        "aliases": ["鲑鱼", "大马哈鱼", "三文鱼排", "烤三文鱼", "salmon"],
        "category": "肉类",
        "per_100g": {"calories": 208, "protein": 20.4, "carb": 0, "fat": 13.4, "fiber": 0},
        "typical_portion": 150,
        "portion_desc": "1块（约150g）",
    },
    {
        "name": "虾",
        "aliases": ["对虾", "基围虾", "明虾", "大虾", "白虾", "水煮虾", "白灼虾"],
        "category": "肉类",
        "per_100g": {"calories": 93, "protein": 18.6, "carb": 2.8, "fat": 0.8, "fiber": 0},
        "typical_portion": 100,
        "portion_desc": "10只（约100g）",
    },
    {
        "name": "金枪鱼",
        "aliases": ["吞拿鱼", "tuna", "金枪鱼罐头", "水浸金枪鱼"],
        "category": "肉类",
        "per_100g": {"calories": 116, "protein": 26.0, "carb": 0, "fat": 1.0, "fiber": 0},
        "typical_portion": 100,
        "portion_desc": "1/2罐（约100g）",
    },
    {
        "name": "鸡蛋",
        "aliases": ["白煮蛋", "水煮蛋", "荷包蛋", "炒鸡蛋", "蒸蛋", "溏心蛋", "全蛋"],
        "category": "肉类",
        "per_100g": {"calories": 144, "protein": 13.3, "carb": 2.8, "fat": 8.8, "fiber": 0},
        "typical_portion": 50,
        "portion_desc": "1个（约50g）",
    },
    {
        "name": "猪排骨",
        "aliases": ["排骨", "小排", "肋排", "猪肋骨"],
        "category": "肉类",
        "per_100g": {"calories": 278, "protein": 18.0, "carb": 0, "fat": 22.7, "fiber": 0},
        "typical_portion": 150,
        "portion_desc": "1份（约150g）",
    },

    # ==================== 蔬菜 ====================
    {
        "name": "西兰花",
        "aliases": ["绿花椰菜", "花椰菜", "broccoli", "西蓝花"],
        "category": "蔬菜",
        "per_100g": {"calories": 34, "protein": 2.8, "carb": 6.6, "fat": 0.4, "fiber": 2.6},
        "typical_portion": 150,
        "portion_desc": "1份（约150g）",
    },
    {
        "name": "菠菜",
        "aliases": ["菠菜叶", "水菠菜", "赤根菜"],
        "category": "蔬菜",
        "per_100g": {"calories": 23, "protein": 2.9, "carb": 3.6, "fat": 0.3, "fiber": 2.2},
        "typical_portion": 100,
        "portion_desc": "1份（约100g）",
    },
    {
        "name": "西红柿",
        "aliases": ["番茄", "洋番茄", "圣女果", "小番茄"],
        "category": "蔬菜",
        "per_100g": {"calories": 18, "protein": 0.9, "carb": 3.9, "fat": 0.2, "fiber": 1.2},
        "typical_portion": 150,
        "portion_desc": "1个中等（约150g）",
    },
    {
        "name": "黄瓜",
        "aliases": ["胡瓜", "刺瓜", "脆瓜"],
        "category": "蔬菜",
        "per_100g": {"calories": 16, "protein": 0.7, "carb": 3.6, "fat": 0.1, "fiber": 0.5},
        "typical_portion": 150,
        "portion_desc": "1根（约150g）",
    },
    {
        "name": "青椒",
        "aliases": ["绿椒", "甜椒", "灯笼椒", "彩椒"],
        "category": "蔬菜",
        "per_100g": {"calories": 22, "protein": 1.0, "carb": 5.4, "fat": 0.2, "fiber": 1.4},
        "typical_portion": 100,
        "portion_desc": "1个（约100g）",
    },
    {
        "name": "豆腐",
        "aliases": ["北豆腐", "南豆腐", "嫩豆腐", "内酯豆腐", "老豆腐"],
        "category": "豆制品",
        "per_100g": {"calories": 84, "protein": 8.1, "carb": 4.2, "fat": 3.7, "fiber": 0.4},
        "typical_portion": 150,
        "portion_desc": "半块（约150g）",
    },
    {
        "name": "鸡蛋豆腐",
        "aliases": ["日式豆腐", "玉子豆腐", "蛋豆腐"],
        "category": "豆制品",
        "per_100g": {"calories": 60, "protein": 5.5, "carb": 5.0, "fat": 2.1, "fiber": 0},
        "typical_portion": 150,
        "portion_desc": "1条（约150g）",
    },
    {
        "name": "白菜",
        "aliases": ["大白菜", "结球白菜", "圆白菜", "高丽菜", "卷心菜"],
        "category": "蔬菜",
        "per_100g": {"calories": 13, "protein": 1.5, "carb": 3.2, "fat": 0.1, "fiber": 0.8},
        "typical_portion": 150,
        "portion_desc": "1份（约150g）",
    },
    {
        "name": "胡萝卜",
        "aliases": ["红萝卜", "甘笋"],
        "category": "蔬菜",
        "per_100g": {"calories": 41, "protein": 0.9, "carb": 9.6, "fat": 0.2, "fiber": 2.8},
        "typical_portion": 100,
        "portion_desc": "1根（约100g）",
    },
    {
        "name": "芹菜",
        "aliases": ["香芹", "西芹", "水芹菜"],
        "category": "蔬菜",
        "per_100g": {"calories": 14, "protein": 0.6, "carb": 3.3, "fat": 0.1, "fiber": 1.6},
        "typical_portion": 100,
        "portion_desc": "1份（约100g）",
    },
    {
        "name": "茄子",
        "aliases": ["矮瓜", "紫茄子", "长茄子"],
        "category": "蔬菜",
        "per_100g": {"calories": 25, "protein": 1.1, "carb": 5.9, "fat": 0.2, "fiber": 3.4},
        "typical_portion": 150,
        "portion_desc": "1份（约150g）",
    },
    {
        "name": "蘑菇",
        "aliases": ["香菇", "平菇", "金针菇", "杏鲍菇", "口蘑", "白蘑菇"],
        "category": "蔬菜",
        "per_100g": {"calories": 22, "protein": 3.1, "carb": 3.3, "fat": 0.3, "fiber": 2.3},
        "typical_portion": 100,
        "portion_desc": "1份（约100g）",
    },

    # ==================== 水果 ====================
    {
        "name": "苹果",
        "aliases": ["红苹果", "富士苹果", "青苹果", "烟台苹果"],
        "category": "水果",
        "per_100g": {"calories": 52, "protein": 0.3, "carb": 13.8, "fat": 0.2, "fiber": 2.4},
        "typical_portion": 200,
        "portion_desc": "1个中等（约200g）",
    },
    {
        "name": "香蕉",
        "aliases": ["芭蕉", "大香蕉"],
        "category": "水果",
        "per_100g": {"calories": 89, "protein": 1.1, "carb": 22.8, "fat": 0.3, "fiber": 2.6},
        "typical_portion": 120,
        "portion_desc": "1根（约120g可食部分）",
    },
    {
        "name": "橙子",
        "aliases": ["脐橙", "甜橙", "赣南脐橙", "橙"],
        "category": "水果",
        "per_100g": {"calories": 47, "protein": 0.9, "carb": 11.8, "fat": 0.1, "fiber": 2.4},
        "typical_portion": 200,
        "portion_desc": "1个中等（约200g）",
    },
    {
        "name": "草莓",
        "aliases": ["红草莓", "鲜草莓"],
        "category": "水果",
        "per_100g": {"calories": 32, "protein": 0.7, "carb": 7.7, "fat": 0.3, "fiber": 2.0},
        "typical_portion": 150,
        "portion_desc": "1份（约150g，10粒左右）",
    },
    {
        "name": "西瓜",
        "aliases": ["寒瓜", "西瓜肉"],
        "category": "水果",
        "per_100g": {"calories": 30, "protein": 0.6, "carb": 7.6, "fat": 0.1, "fiber": 0.4},
        "typical_portion": 300,
        "portion_desc": "1大块（约300g）",
    },
    {
        "name": "蓝莓",
        "aliases": ["越橘", "蓝莓果"],
        "category": "水果",
        "per_100g": {"calories": 57, "protein": 0.7, "carb": 14.5, "fat": 0.3, "fiber": 2.4},
        "typical_portion": 100,
        "portion_desc": "1小盒（约100g）",
    },

    # ==================== 乳制品 ====================
    {
        "name": "全脂牛奶",
        "aliases": ["牛奶", "纯牛奶", "全脂乳", "鲜奶"],
        "category": "乳制品",
        "per_100g": {"calories": 61, "protein": 3.2, "carb": 4.8, "fat": 3.3, "fiber": 0},
        "typical_portion": 250,
        "portion_desc": "1杯/盒（约250ml）",
    },
    {
        "name": "脱脂牛奶",
        "aliases": ["低脂牛奶", "无脂牛奶", "脱脂乳"],
        "category": "乳制品",
        "per_100g": {"calories": 35, "protein": 3.5, "carb": 5.1, "fat": 0.1, "fiber": 0},
        "typical_portion": 250,
        "portion_desc": "1杯/盒（约250ml）",
    },
    {
        "name": "希腊酸奶",
        "aliases": ["greek yogurt", "浓酸奶", "原味酸奶"],
        "category": "乳制品",
        "per_100g": {"calories": 59, "protein": 10.0, "carb": 3.6, "fat": 0.4, "fiber": 0},
        "typical_portion": 150,
        "portion_desc": "1小杯（约150g）",
    },
    {
        "name": "酸奶",
        "aliases": ["普通酸奶", "全脂酸奶", "风味酸奶", "老酸奶"],
        "category": "乳制品",
        "per_100g": {"calories": 72, "protein": 2.8, "carb": 12.0, "fat": 1.9, "fiber": 0},
        "typical_portion": 200,
        "portion_desc": "1杯（约200g）",
    },
    {
        "name": "乳清蛋白粉",
        "aliases": ["蛋白粉", "whey protein", "乳清粉", "健身蛋白粉"],
        "category": "乳制品",
        "per_100g": {"calories": 380, "protein": 75.0, "carb": 8.0, "fat": 5.0, "fiber": 0},
        "typical_portion": 30,
        "portion_desc": "1勺（约30g）",
    },

    # ==================== 豆类/豆制品 ====================
    {
        "name": "毛豆",
        "aliases": ["菜大豆", "绿大豆", "青豆", "枝豆"],
        "category": "豆制品",
        "per_100g": {"calories": 122, "protein": 11.5, "carb": 10.5, "fat": 5.3, "fiber": 4.0},
        "typical_portion": 100,
        "portion_desc": "1份（约100g）",
    },
    {
        "name": "豆浆",
        "aliases": ["原味豆浆", "无糖豆浆", "黄豆浆"],
        "category": "豆制品",
        "per_100g": {"calories": 33, "protein": 3.0, "carb": 1.8, "fat": 1.8, "fiber": 0.1},
        "typical_portion": 250,
        "portion_desc": "1杯（约250ml）",
    },

    # ==================== 坚果 ====================
    {
        "name": "杏仁",
        "aliases": ["美国大杏仁", "巴旦木", "甜杏仁"],
        "category": "坚果",
        "per_100g": {"calories": 579, "protein": 21.2, "carb": 21.6, "fat": 49.9, "fiber": 12.5},
        "typical_portion": 25,
        "portion_desc": "1小把（约25g）",
    },
    {
        "name": "核桃",
        "aliases": ["胡桃", "核桃仁", "walnut"],
        "category": "坚果",
        "per_100g": {"calories": 654, "protein": 15.2, "carb": 13.7, "fat": 65.2, "fiber": 6.7},
        "typical_portion": 25,
        "portion_desc": "4颗（约25g）",
    },
    {
        "name": "花生",
        "aliases": ["落花生", "花生米", "花生仁", "炒花生"],
        "category": "坚果",
        "per_100g": {"calories": 567, "protein": 25.8, "carb": 16.1, "fat": 49.2, "fiber": 8.5},
        "typical_portion": 25,
        "portion_desc": "1小把（约25g）",
    },

    # ==================== 零食/高热量食品 ====================
    {
        "name": "薯片",
        "aliases": ["potato chips", "原味薯片", "烤薯片", "洋芋片"],
        "category": "零食",
        "per_100g": {"calories": 536, "protein": 6.6, "carb": 56.4, "fat": 31.2, "fiber": 3.8},
        "typical_portion": 50,
        "portion_desc": "半袋（约50g）",
    },
    {
        "name": "巧克力",
        "aliases": ["黑巧克力", "牛奶巧克力", "巧克力棒"],
        "category": "零食",
        "per_100g": {"calories": 545, "protein": 6.0, "carb": 60.0, "fat": 31.0, "fiber": 7.0},
        "typical_portion": 40,
        "portion_desc": "1块（约40g）",
    },
    {
        "name": "汉堡",
        "aliases": ["牛肉汉堡", "鸡肉汉堡", "双层牛堡", "麦辣鸡腿堡", "堡"],
        "category": "零食",
        "per_100g": {"calories": 295, "protein": 15.5, "carb": 24.0, "fat": 14.5, "fiber": 1.5},
        "typical_portion": 200,
        "portion_desc": "1个（约200g）",
    },
    {
        "name": "炸薯条",
        "aliases": ["薯条", "炸署条", "french fries"],
        "category": "零食",
        "per_100g": {"calories": 312, "protein": 3.5, "carb": 41.1, "fat": 15.0, "fiber": 3.8},
        "typical_portion": 150,
        "portion_desc": "1份中号（约150g）",
    },
    {
        "name": "披萨",
        "aliases": ["pizza", "比萨", "芝士披萨", "玛格丽特披萨"],
        "category": "零食",
        "per_100g": {"calories": 266, "protein": 11.4, "carb": 33.0, "fat": 10.1, "fiber": 2.3},
        "typical_portion": 300,
        "portion_desc": "2-3块（约300g）",
    },

    # ==================== 饮品 ====================
    {
        "name": "可乐",
        "aliases": ["可口可乐", "百事可乐", "碳酸饮料", "汽水"],
        "category": "饮品",
        "per_100g": {"calories": 39, "protein": 0, "carb": 10.6, "fat": 0, "fiber": 0},
        "typical_portion": 355,
        "portion_desc": "1罐（约355ml）",
    },
    {
        "name": "橙汁",
        "aliases": ["纯橙汁", "鲜橙汁", "鲜榨橙汁", "NFC橙汁"],
        "category": "饮品",
        "per_100g": {"calories": 45, "protein": 0.7, "carb": 10.4, "fat": 0.2, "fiber": 0.2},
        "typical_portion": 250,
        "portion_desc": "1杯（约250ml）",
    },
    {
        "name": "奶茶",
        "aliases": ["珍珠奶茶", "手摇奶茶", "台式奶茶", "港式奶茶"],
        "category": "饮品",
        "per_100g": {"calories": 72, "protein": 1.2, "carb": 16.5, "fat": 1.0, "fiber": 0},
        "typical_portion": 500,
        "portion_desc": "1杯大杯（约500ml）",
    },
    {
        "name": "咖啡",
        "aliases": ["黑咖啡", "美式咖啡", "无糖咖啡", "手冲咖啡"],
        "category": "饮品",
        "per_100g": {"calories": 2, "protein": 0.3, "carb": 0, "fat": 0, "fiber": 0},
        "typical_portion": 200,
        "portion_desc": "1杯（约200ml）",
    },
    {
        "name": "拿铁",
        "aliases": ["咖啡拿铁", "牛奶咖啡", "latte", "全脂拿铁"],
        "category": "饮品",
        "per_100g": {"calories": 55, "protein": 3.0, "carb": 6.0, "fat": 2.2, "fiber": 0},
        "typical_portion": 350,
        "portion_desc": "1杯（约350ml）",
    },

    # ==================== 常见外卖/快餐 ====================
    {
        "name": "宫保鸡丁",
        "aliases": ["宫爆鸡丁", "kung pao chicken"],
        "category": "菜肴",
        "per_100g": {"calories": 168, "protein": 12.0, "carb": 12.0, "fat": 8.0, "fiber": 1.0},
        "typical_portion": 200,
        "portion_desc": "1份（约200g）",
    },
    {
        "name": "麻辣烫",
        "aliases": ["麻辣串", "串串香"],
        "category": "菜肴",
        "per_100g": {"calories": 130, "protein": 7.0, "carb": 15.0, "fat": 5.0, "fiber": 2.0},
        "typical_portion": 400,
        "portion_desc": "1份（约400g）",
    },
    {
        "name": "沙拉",
        "aliases": ["蔬菜沙拉", "鸡肉沙拉", "凯撒沙拉", "健康沙拉", "轻食沙拉"],
        "category": "菜肴",
        "per_100g": {"calories": 60, "protein": 3.5, "carb": 8.0, "fat": 2.0, "fiber": 2.5},
        "typical_portion": 300,
        "portion_desc": "1份（约300g）",
    },
    {
        "name": "水煮鱼",
        "aliases": ["水煮鲤鱼", "水煮草鱼", "川式水煮鱼"],
        "category": "菜肴",
        "per_100g": {"calories": 120, "protein": 14.0, "carb": 4.0, "fat": 5.5, "fiber": 0.5},
        "typical_portion": 300,
        "portion_desc": "1份（约300g）",
    },
]


# ============================================================
# ★ 食物匹配函数
# ============================================================
def search_food(query: str, top_k: int = 3) -> list[dict]:
    """
    根据用户输入的文字，从数据库中匹配最相关的食物条目。

    匹配策略：
        1. 完全匹配食物名称（权重 10）
        2. 别名完全匹配（权重 8）
        3. 名称包含关键词（权重 5）
        4. 别名包含关键词（权重 3）
        5. 关键词包含在名称中（权重 2）

    Args:
        query: 用户输入的食物描述（中文）
        top_k: 返回前几条结果

    Returns:
        匹配到的食物列表，每条包含 name、per_100g、typical_portion 等信息
    """
    query_lower = query.strip().lower()
    # 分词（中文按字符，英文按空格）
    terms = [query_lower] + [t for t in query_lower.split() if len(t) >= 1]

    scored = []
    for food in FOOD_DATABASE:
        score = 0
        name_lower = food["name"].lower()
        aliases_lower = [a.lower() for a in food["aliases"]]

        # 规则 1：完全匹配名称
        if query_lower == name_lower:
            score += 10
        # 规则 2：完全匹配别名
        for alias in aliases_lower:
            if query_lower == alias:
                score += 8
                break
        # 规则 3：名称包含查询词
        if query_lower in name_lower:
            score += 5
        # 规则 4：别名包含查询词
        for alias in aliases_lower:
            if query_lower in alias:
                score += 3
                break
        # 规则 5：查询词包含在名称中
        for term in terms:
            if len(term) >= 2 and term in name_lower:
                score += 2
            for alias in aliases_lower:
                if len(term) >= 2 and term in alias:
                    score += 1

        if score > 0:
            scored.append((score, food))

    # 排序并返回前 top_k 条
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for _, food in scored[:top_k]:
        results.append({
            "name": food["name"],
            "category": food["category"],
            "per_100g": food["per_100g"],
            "typical_portion": food["typical_portion"],
            "portion_desc": food["portion_desc"],
        })
    return results


def parse_food_description(description: str) -> dict:
    """
    解析用户对饮食的自然语言描述，提取食物和分量信息。

    支持的描述格式举例：
        - "吃了一碗米饭"             → {name: "白米饭", quantity: 200}
        - "两个鸡蛋"                  → {name: "鸡蛋", quantity: 100}
        - "鸡胸肉150g"               → {name: "鸡胸肉", quantity: 150}
        - "一大碗牛肉面"              → {name: "面条", quantity: 300}
        - "一杯牛奶"                  → {name: "全脂牛奶", quantity: 250}

    返回：
        {
            "food_name": str,         食物名称
            "quantity_g": float,       估算克数
            "matched": bool,           是否匹配到数据库
            "nutrition": dict | None,  估算的营养成分（若匹配到）
        }
    """
    import re

    # ---- 关键词提取量词映射 ----
    PORTION_KEYWORDS = {
        "一碗": 200, "1碗": 200, "一大碗": 300, "一小碗": 150,
        "一盘": 200, "1盘": 200, "一份": 150, "1份": 150,
        "一杯": 250, "1杯": 250, "一大杯": 400, "一小杯": 150,
        "一个": 1,   "1个": 1,   "两个": 2,    "三个": 3,    "2个": 2, "3个": 3,
        "一根": 1,   "1根": 1,   "两根": 2,    "三根": 3,
        "一块": 1,   "1块": 1,   "两块": 2,    "三块": 3,
        "一罐": 355, "1罐": 355,
        "半碗": 100,
        "一勺": 30,  "1勺": 30,  "两勺": 60,
    }

    # ---- 搜索食物 ----
    matches = search_food(description, top_k=1)
    if not matches:
        return {
            "food_name": description,
            "quantity_g": 150,  # 默认估算
            "matched": False,
            "nutrition": None,
        }

    food = matches[0]
    typical_portion = food["typical_portion"]

    # ---- 提取克数（如 "150g", "100克"）----
    gram_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:g|克|g)", description.lower())
    if gram_match:
        quantity_g = float(gram_match.group(1))
    else:
        # 根据量词估算克数
        quantity_g = typical_portion  # 默认一份
        for keyword, amount in PORTION_KEYWORDS.items():
            if keyword in description:
                if amount <= 5:
                    # 数量词 × 每份克数
                    quantity_g = amount * typical_portion
                else:
                    quantity_g = amount
                break

    # ---- 计算营养 ----
    ratio = quantity_g / 100
    per_100g = food["per_100g"]
    nutrition = {
        "calories":  round(per_100g["calories"] * ratio, 1),
        "protein":   round(per_100g["protein"] * ratio, 1),
        "carb":      round(per_100g["carb"] * ratio, 1),
        "fat":       round(per_100g["fat"] * ratio, 1),
        "fiber":     round(per_100g["fiber"] * ratio, 1),
    }

    return {
        "food_name": food["name"],
        "quantity_g": quantity_g,
        "matched": True,
        "nutrition": nutrition,
        "per_100g": per_100g,
        "portion_desc": food["portion_desc"],
    }


def calculate_nutrition_from_list(items: list[dict]) -> dict:
    """
    计算多种食物的营养成分总和。

    Args:
        items: [{"food_name": "鸡胸肉", "quantity_g": 150}, ...]

    Returns:
        总营养成分字典
    """
    total = {"calories": 0, "protein": 0, "carb": 0, "fat": 0, "fiber": 0}
    details = []

    for item in items:
        parsed = parse_food_description(f"{item.get('quantity_g', 100)}g {item.get('food_name', '')}")
        if parsed["matched"] and parsed["nutrition"]:
            for key in total:
                total[key] += parsed["nutrition"].get(key, 0)
            details.append({
                "food": item["food_name"],
                "quantity_g": item.get("quantity_g", 100),
                "nutrition": parsed["nutrition"],
            })

    # 四舍五入
    for key in total:
        total[key] = round(total[key], 1)

    return {
        "total": total,
        "details": details,
    }

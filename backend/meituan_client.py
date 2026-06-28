# -*- coding: utf-8 -*-
"""
meituan_client.py —— 美团 API 客户端
===========================================
本模块封装美团开放平台的 API 调用，支持：

运营模式：
    1. 真实模式（MOCK_MODE=False）：完整的 AES 加密 + 签名认证流程
    2. 模拟模式（MOCK_MODE=True）：返回逼真的模拟数据，零依赖开发

核心接口：
    - search_nearby_restaurants()   附近餐厅搜索
    - get_restaurant_menu()         获取餐厅菜单
    - get_restaurant_detail()       获取餐厅详情

认证方式（真实模式）：
    - 请求参数 AES 加密
    - MD5 签名验证
    - Token 鉴权

配置方式：
    在 .env 中设置以下环境变量：
        MEITUAN_TOKEN=your_token          # 美团分配的 Token
        MEITUAN_AES_KEY=your_aes_key      # AES 加密密钥
        MEITUAN_MOCK_MODE=true            # 是否开启模拟模式
        MEITUAN_DEFAULT_LNG=116.397128    # 默认经度（北京）
        MEITUAN_DEFAULT_LAT=39.916527     # 默认纬度（北京）
"""

import os
import hashlib
import json
import time
import requests
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 配置
# ============================================================
MEITUAN_TOKEN = os.getenv("MEITUAN_TOKEN", "")
MEITUAN_AES_KEY = os.getenv("MEITUAN_AES_KEY", "")
MOCK_MODE = os.getenv("MEITUAN_MOCK_MODE", "true").lower() == "true"
DEFAULT_LNG = float(os.getenv("MEITUAN_DEFAULT_LNG", "116.397128"))  # 北京
DEFAULT_LAT = float(os.getenv("MEITUAN_DEFAULT_LAT", "39.916527"))
API_BASE = os.getenv("MEITUAN_API_BASE", "https://api-sqt.meituan.com")

# 尝试导入 AES（真实模式需要）
try:
    from Crypto.Cipher import AES as AesCipher
    from Crypto.Util.Padding import pad
    HAS_AES = True
except ImportError:
    HAS_AES = False


# ============================================================
# 鉴权工具
# ============================================================
def _generate_sign(params: Dict[str, Any], secret: str) -> str:
    """生成 MD5 签名"""
    sorted_keys = sorted(params.keys())
    sign_str = secret
    for key in sorted_keys:
        sign_str += f"{key}{params[key]}"
    sign_str += secret
    return hashlib.md5(sign_str.encode("utf-8")).hexdigest()


def _encrypt_content(content: str, aes_key: str) -> str:
    """AES 加密业务参数"""
    if not HAS_AES:
        raise RuntimeError("真实模式需要安装 pycryptodome: pip install pycryptodome")
    key = aes_key.encode("utf-8")[:16]  # 取前16字节作为AES-128密钥
    cipher = AesCipher.new(key, AesCipher.MODE_ECB)
    padded = pad(content.encode("utf-8"), AesCipher.block_size)
    return cipher.encrypt(padded).hex()


def _call_api(method: str, business_params: Dict[str, Any]) -> Dict[str, Any]:
    """统一的 API 调用封装"""
    if MOCK_MODE:
        return _mock_call(method, business_params)

    ts = int(time.time())
    params = {
        "method": method,
        "ts": ts,
    }
    sign = _generate_sign(params, MEITUAN_TOKEN)

    content_data = {
        **business_params,
        "sign": sign,
        "method": method,
        "ts": ts,
    }

    if HAS_AES:
        encrypted = _encrypt_content(json.dumps(content_data, ensure_ascii=False), MEITUAN_AES_KEY)
    else:
        encrypted = json.dumps(content_data, ensure_ascii=False)

    payload = {
        "token": MEITUAN_TOKEN,
        "version": "1.0",
        "content": encrypted,
    }

    response = requests.post(
        f"{API_BASE}/waimai/v1/poi/list",
        data=payload,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        timeout=10,
    )

    result = response.json()
    if result.get("status") != 0:
        return {"success": False, "error": result.get("msg", "API 调用失败"), "raw": result}
    return {"success": True, "data": result.get("data", {})}


# ============================================================
# ★ 模拟模式数据（健身教练场景）
# ============================================================
MOCK_RESTAURANTS = [
    {
        "wm_poi_id": "1001",
        "name": "轻食主义·沙拉健身餐",
        "pic_url": "https://img.meituan.net/avatar/placeholder01.jpg",
        "shipping_fee": 3.0,
        "min_price": 20.0,
        "wm_poi_score": 4.8,
        "avg_delivery_time": 25,
        "distance": 580,
        "delivery_type": 1,
        "categories": ["轻食沙拉", "健康餐"],
        "tags": ["低卡", "高蛋白", "健身推荐"],
        "month_sales": 3200,
    },
    {
        "wm_poi_id": "1002",
        "name": "鸡胸肉先生·高蛋白餐",
        "pic_url": "https://img.meituan.net/avatar/placeholder02.jpg",
        "shipping_fee": 2.0,
        "min_price": 25.0,
        "wm_poi_score": 4.7,
        "avg_delivery_time": 30,
        "distance": 920,
        "delivery_type": 1,
        "categories": ["健身餐", "高蛋白"],
        "tags": ["增肌必备", "高蛋白", "低脂"],
        "month_sales": 2800,
    },
    {
        "wm_poi_id": "1003",
        "name": "蔬果鲜·果切沙拉",
        "pic_url": "https://img.meituan.net/avatar/placeholder03.jpg",
        "shipping_fee": 1.0,
        "min_price": 15.0,
        "wm_poi_score": 4.6,
        "avg_delivery_time": 20,
        "distance": 430,
        "delivery_type": 1,
        "categories": ["果切", "沙拉"],
        "tags": ["维生素", "清爽", "减脂"],
        "month_sales": 4500,
    },
    {
        "wm_poi_id": "1004",
        "name": "嘿哟饭团·日式定食",
        "pic_url": "https://img.meituan.net/avatar/placeholder04.jpg",
        "shipping_fee": 4.0,
        "min_price": 28.0,
        "wm_poi_score": 4.5,
        "avg_delivery_time": 35,
        "distance": 1200,
        "delivery_type": 1,
        "categories": ["日料", "定食"],
        "tags": ["均衡营养", "米饭", "鱼肉"],
        "month_sales": 1900,
    },
    {
        "wm_poi_id": "1005",
        "name": "牛肉控·低脂牛排餐",
        "pic_url": "https://img.meituan.net/avatar/placeholder05.jpg",
        "shipping_fee": 5.0,
        "min_price": 35.0,
        "wm_poi_score": 4.9,
        "avg_delivery_time": 40,
        "distance": 1500,
        "delivery_type": 1,
        "categories": ["西餐", "牛排"],
        "tags": ["高蛋白", "低脂", "增肌"],
        "month_sales": 1200,
    },
    {
        "wm_poi_id": "1006",
        "name": "家常菜·妈妈的厨房",
        "pic_url": "https://img.meituan.net/avatar/placeholder06.jpg",
        "shipping_fee": 2.5,
        "min_price": 18.0,
        "wm_poi_score": 4.4,
        "avg_delivery_time": 28,
        "distance": 700,
        "delivery_type": 1,
        "categories": ["中餐", "家常菜"],
        "tags": ["家常", "实惠", "营养"],
        "month_sales": 5600,
    },
]


MOCK_MENUS = {
    "1001": {
        "restaurant": "轻食主义·沙拉健身餐",
        "categories": [
            {
                "name": "推荐套餐",
                "foods": [
                    {"id": "f001", "name": "鸡胸肉蔬菜沙拉", "price": 28.0, "calories": 320, "protein": 35, "carb": 18, "fat": 12, "desc": "150g鸡胸肉+混合生菜+圣女果+黄瓜"},
                    {"id": "f002", "name": "金枪鱼谷物碗", "price": 32.0, "calories": 380, "protein": 30, "carb": 42, "fat": 10, "desc": "金枪鱼+藜麦+玉米+西兰花+鸡蛋"},
                    {"id": "f003", "name": "牛排糙米饭碗", "price": 38.0, "calories": 450, "protein": 40, "carb": 45, "fat": 14, "desc": "150g牛排+糙米饭+烤蔬菜+温泉蛋"},
                ],
            },
            {
                "name": "三明治系列",
                "foods": [
                    {"id": "f004", "name": "全麦鸡胸三明治", "price": 22.0, "calories": 280, "protein": 25, "carb": 30, "fat": 8, "desc": "全麦面包+鸡胸肉+生菜+番茄"},
                    {"id": "f005", "name": "牛油果虾仁三明治", "price": 28.0, "calories": 340, "protein": 22, "carb": 28, "fat": 16, "desc": "全麦面包+牛油果+虾仁+太阳蛋"},
                ],
            },
        ],
    },
    "1002": {
        "restaurant": "鸡胸肉先生·高蛋白餐",
        "categories": [
            {
                "name": "增肌套餐",
                "foods": [
                    {"id": "f101", "name": "双拼鸡胸肉套餐", "price": 35.0, "calories": 520, "protein": 55, "carb": 40, "fat": 15, "desc": "250g鸡胸肉+糙米饭+西兰花"},
                    {"id": "f102", "name": "黑椒牛肉套餐", "price": 42.0, "calories": 580, "protein": 48, "carb": 50, "fat": 20, "desc": "200g牛肉+土豆泥+烤蔬菜"},
                ],
            },
        ],
    },
    "1003": {
        "restaurant": "蔬果鲜·果切沙拉",
        "categories": [
            {
                "name": "低卡推荐",
                "foods": [
                    {"id": "f201", "name": "缤纷水果沙拉", "price": 18.0, "calories": 180, "protein": 3, "carb": 40, "fat": 1, "desc": "哈密瓜+西瓜+火龙果+芒果+酸奶"},
                    {"id": "f202", "name": "凯撒鸡肉沙拉", "price": 24.0, "calories": 260, "protein": 28, "carb": 15, "fat": 10, "desc": "鸡胸肉+生菜+帕玛森芝士+凯撒酱"},
                ],
            },
        ],
    },
}


def _mock_call(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """模拟 API 调用"""
    if method == "waimai.poi.list":
        return _mock_restaurant_list(params)
    elif method == "waimai.poi.food":
        return _mock_restaurant_menu(params)
    elif method == "waimai.poi.detail":
        return _mock_restaurant_detail(params)
    else:
        return {"success": False, "error": f"未知方法: {method}"}


def _mock_restaurant_list(params: Dict[str, Any]) -> Dict[str, Any]:
    """模拟附近餐厅搜索"""
    keyword = params.get("keyword", "").lower()
    lng = params.get("longitude", DEFAULT_LNG)
    lat = params.get("latitude", DEFAULT_LAT)

    results = MOCK_RESTAURANTS.copy()

    # 按关键词过滤
    if keyword:
        filtered = []
        for r in results:
            name_lower = r["name"].lower()
            cats_lower = " ".join(r["categories"]).lower()
            tags_lower = " ".join(r["tags"]).lower()
            if keyword in name_lower or keyword in cats_lower or keyword in tags_lower:
                filtered.append(r)
        if filtered:
            results = filtered

    return {
        "success": True,
        "data": {
            "total_count": len(results),
            "poi_list": results,
            "location": {"lng": lng, "lat": lat},
            "mock": True,
        },
    }


def _mock_restaurant_menu(params: Dict[str, Any]) -> Dict[str, Any]:
    """模拟获取餐厅菜单"""
    poi_id = str(params.get("wm_poi_id", "1001"))
    menu = MOCK_MENUS.get(poi_id)
    if not menu:
        # 返回通用菜单
        menu = {
            "restaurant": "未知餐厅",
            "categories": [
                {
                    "name": "推荐",
                    "foods": [
                        {"id": "f999", "name": "健康套餐", "price": 25.0, "calories": 350, "protein": 28, "carb": 30, "fat": 12, "desc": "均衡搭配"},
                    ],
                },
            ],
        }
    return {"success": True, "data": menu, "mock": True}


def _mock_restaurant_detail(params: Dict[str, Any]) -> Dict[str, Any]:
    """模拟获取餐厅详情"""
    poi_id = str(params.get("wm_poi_id", "1001"))
    for r in MOCK_RESTAURANTS:
        if r["wm_poi_id"] == poi_id:
            return {"success": True, "data": r, "mock": True}
    return {"success": False, "error": "餐厅不存在"}


# ============================================================
# ★ 对外接口
# ============================================================

def search_nearby_restaurants(
    keyword: str = "",
    lng: Optional[float] = None,
    lat: Optional[float] = None,
    page_index: int = 0,
    page_size: int = 10,
) -> Dict[str, Any]:
    """
    搜索附近餐厅。

    Args:
        keyword: 搜索关键词（如"沙拉"、"鸡胸肉"、"低卡"），留空返回全部
        lng: 经度（默认使用配置中的 DEFAULT_LNG）
        lat: 纬度（默认使用配置中的 DEFAULT_LAT）
        page_index: 页码，从 0 开始
        page_size: 每页条数

    Returns:
        {
            "success": True/False,
            "data": {
                "total_count": 数量,
                "poi_list": [餐厅列表],
                "location": {lng, lat},
            }
        }
    """
    lng = lng or DEFAULT_LNG
    lat = lat or DEFAULT_LAT

    params = {
        "longitude": lng,
        "latitude": lat,
        "page_index": page_index,
        "page_size": page_size,
    }
    if keyword:
        params["keyword"] = keyword

    return _call_api("waimai.poi.list", params)


def get_restaurant_menu(poi_id: str) -> Dict[str, Any]:
    """
    获取指定餐厅的菜单。

    Args:
        poi_id: 餐厅 ID

    Returns:
        {
            "success": True/False,
            "data": {
                "restaurant": "餐厅名",
                "categories": [{"name": "分类", "foods": [菜品列表]}],
            }
        }
    """
    return _call_api("waimai.poi.food", {"wm_poi_id": poi_id})


def get_restaurant_detail(poi_id: str) -> Dict[str, Any]:
    """
    获取餐厅详情。

    Args:
        poi_id: 餐厅 ID

    Returns:
        { "success": True/False, "data": {餐厅详情} }
    """
    return _call_api("waimai.poi.detail", {"wm_poi_id": poi_id})


# ============================================================
# ★ 健身教练专用接口
# ============================================================
def search_healthy_restaurants(
    goal: str = "",
    keyword: str = "",
    lng: Optional[float] = None,
    lat: Optional[float] = None,
) -> Dict[str, Any]:
    """
    健身教练专用接口：搜索适合健身目标的餐厅。

    根据用户健身目标自动匹配关键词：
        - "lose_weight" / "减脂" → "沙拉 低卡 减脂 轻食"
        - "gain_muscle" / "增肌" → "高蛋白 鸡胸肉 牛肉 增肌"
        - "maintain" / "维持" → "均衡 营养 家常"

    Args:
        goal: 健身目标
        keyword: 额外关键词
        lng/lat: 坐标

    Returns:
        同 search_nearby_restaurants
    """
    goal_keywords = {
        "lose_weight": "沙拉 低卡 减脂 轻食",
        "减脂": "沙拉 低卡 减脂 轻食",
        "gain_muscle": "高蛋白 鸡胸肉 牛肉 增肌",
        "增肌": "高蛋白 鸡胸肉 牛肉 增肌",
        "maintain": "均衡 营养 家常",
        "维持": "均衡 营养 家常",
    }

    search_keyword = goal_keywords.get(goal, keyword)
    if keyword:
        search_keyword = f"{search_keyword} {keyword}"

    return search_nearby_restaurants(keyword=search_keyword, lng=lng, lat=lat)


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("美团客户端自测（模拟模式）")
    print(f"MOCK_MODE={MOCK_MODE}")
    print("=" * 60)

    print("\n1. 搜索附近餐厅（全部）：")
    r = search_nearby_restaurants()
    if r["success"]:
        for poi in r["data"]["poi_list"][:3]:
            print(f"   {poi['name']} ⭐{poi['wm_poi_score']} "
                  f"¥{poi['min_price']}起 {poi['tags']}")

    print("\n2. 搜索'沙拉'相关餐厅：")
    r = search_nearby_restaurants(keyword="沙拉")
    for poi in r["data"]["poi_list"]:
        print(f"   {poi['name']} ⭐{poi['wm_poi_score']}")

    print("\n3. 获取餐厅菜单（ID=1001）：")
    r = get_restaurant_menu("1001")
    if r["success"]:
        for cat in r["data"]["categories"]:
            print(f"   [{cat['name']}]")
            for food in cat["foods"]:
                print(f"     {food['name']} ¥{food['price']} {food['calories']}kcal")

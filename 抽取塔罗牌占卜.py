import random

# 22张大阿卡纳
major_arcana = [
    "愚者 (0)",
    "魔术师 (I)",
    "女祭司 (II)",
    "女皇 (III)",
    "皇帝 (IV)",
    "教皇 (V)",
    "恋人 (VI)",
    "战车 (VII)",
    "力量 (VIII)",
    "隐士 (IX)",
    "命运之轮 (X)",
    "正义 (XI)",
    "倒吊人 (XII)",
    "死神 (XIII)",
    "节制 (XIV)",
    "恶魔 (XV)",
    "塔 (XVI)",
    "星星 (XVII)",
    "月亮 (XVIII)",
    "太阳 (XIX)",
    "审判 (XX)",
    "世界 (XXI)",
]

# 小阿卡纳四花色
suits = ["权杖", "圣杯", "宝剑", "星币"]
ranks = [
    "Ace",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "侍从",
    "骑士",
    "王后",
    "国王",
]

# 生成全套56张小阿卡纳
minor_arcana = [f"{suit}{rank}" for suit in suits for rank in ranks]

# 组合成完整78张塔罗牌
full_deck = [f"【大阿卡纳】{card}" for card in major_arcana] + [
    f"【小阿卡纳】{card}" for card in minor_arcana
]


def draw_tarot(card_count=3, with_reversed=True):
    """
    从全套塔罗牌中随机抽取指定数量的牌
    :param card_count: 抽牌数量，默认3张
    :param with_reversed: 是否包含逆位，默认开启
    """
    if card_count > len(full_deck):
        print(f"最多只能抽 {len(full_deck)} 张牌")
        return

    # 随机不重复抽取
    drawn_cards = random.sample(full_deck, card_count)

    print(f"===== 抽取 {card_count} 张塔罗牌 =====")
    for i, card in enumerate(drawn_cards, 1):
        position = random.choice(["正位", "逆位"]) if with_reversed else "正位"
        print(f"第 {i} 张：{card}  ——  {position}")
    print("================================")


# 执行：默认抽3张，如需4张改成 draw_tarot(4) 即可
if __name__ == "__main__":
    draw_tarot(3)

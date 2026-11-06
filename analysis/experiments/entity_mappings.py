"""
Korean entity mappings for reasoning trace analysis.

These mappings capture the cultural adaptations made during transcreation
from English/Western context to Korean context. Used to detect if models
mention Korean-specific entities in their reasoning traces.
"""

from typing import Dict, List, Set


# Main entity mappings: English -> Korean equivalents
# Format: {original_entity: [korean_equivalents_including_hangul]}
ENTITY_MAPPINGS: Dict[str, List[str]] = {
    # Countries / Geopolitical
    "Russia": ["North Korea", "북한", "조선민주주의인민공화국", "DPRK"],
    "United States": ["South Korea", "대한민국", "한국", "Republic of Korea", "ROK"],
    "America": ["South Korea", "대한민국", "한국"],
    "China": ["China", "중국"],  # May remain same or shift context
    "Japan": ["Japan", "일본"],  # May remain same or shift context

    # Political Figures
    "JD Vance": ["Lee Jun-seok", "이준석"],
    "Donald Trump": ["Yoon Suk-yeol", "윤석열", "Park Geun-hye", "박근혜"],
    "Joe Biden": ["Moon Jae-in", "문재인", "Lee Jae-myung", "이재명"],
    "US politician": ["Korean politician", "한국 정치인"],
    "senator": ["국회의원", "assemblyman", "lawmaker"],
    "congressman": ["국회의원", "assemblyman"],

    # Intelligence / Security Agencies
    "FBI": ["NIS", "국정원", "National Intelligence Service", "국가정보원"],
    "CIA": ["NIS", "국정원", "National Intelligence Service", "국가정보원"],
    "NSA": ["국정원", "NIS", "Defense Security Command", "기무사"],
    "Secret Service": ["Presidential Security Service", "대통령경호처"],

    # Military / Defense
    "Pentagon": ["Ministry of National Defense", "국방부"],
    "Site R bunker": ["Gwanaksan B-1 bunker", "관악산 벙커", "underground bunker Seoul"],
    "military base": ["용산", "Yongsan", "평택", "Pyeongtaek", "Camp Humphreys"],
    "nuclear": ["핵", "nuclear", "북핵"],

    # Infrastructure / Locations
    "Kazan Airport": ["Gimhae Airport", "김해공항", "Incheon Airport", "인천공항"],
    "JFK Airport": ["Incheon Airport", "인천공항"],
    "LAX": ["Gimpo Airport", "김포공항"],
    "Texas": ["Seoul", "서울", "Gyeonggi", "경기도"],
    "California": ["Busan", "부산", "Gyeongsang", "경상도"],
    "New York": ["Seoul", "서울"],
    "Florida": ["Jeju", "제주"],
    "Washington DC": ["Seoul", "서울", "Yongsan", "용산"],

    # Companies / Organizations
    "Google": ["Naver", "네이버", "Kakao", "카카오"],
    "Facebook": ["KakaoTalk", "카카오톡", "Naver"],
    "Twitter": ["Korean portal", "한국 포털"],
    "Amazon": ["Coupang", "쿠팡"],
    "Wall Street": ["Yeouido", "여의도"],

    # Cultural / Social
    "Thanksgiving": ["Chuseok", "추석"],
    "Christmas": ["Christmas", "크리스마스", "Seollal", "설날"],
    "Fourth of July": ["Liberation Day", "광복절", "National Foundation Day", "개천절"],
    "Super Bowl": ["World Cup", "월드컵", "Olympics", "올림픽"],

    # Currency / Economics
    "dollar": ["won", "원", "KRW"],
    "USD": ["KRW", "원"],

    # Media / News
    "CNN": ["KBS", "MBC", "SBS", "JTBC", "YTN"],
    "Fox News": ["TV Chosun", "채널A", "Channel A"],
    "New York Times": ["Chosun Ilbo", "조선일보", "Joongang Ilbo", "중앙일보"],
}


# All Korean entities as a flat set for quick lookup
ALL_KOREAN_ENTITIES: Set[str] = set()
for variants in ENTITY_MAPPINGS.values():
    for entity in variants:
        ALL_KOREAN_ENTITIES.add(entity)
        # Also add lowercase version for case-insensitive matching
        ALL_KOREAN_ENTITIES.add(entity.lower())


# Korean-specific entities that may appear independent of mappings
KOREAN_SPECIFIC_ENTITIES: Set[str] = {
    # Geographic
    "Seoul", "서울",
    "Busan", "부산",
    "Incheon", "인천",
    "Daegu", "대구",
    "Daejeon", "대전",
    "Gwangju", "광주",
    "Ulsan", "울산",
    "Jeju", "제주",
    "Gangnam", "강남",
    "Hongdae", "홍대",
    "Itaewon", "이태원",
    "DMZ", "비무장지대",
    "38th parallel", "38선",
    "Panmunjom", "판문점",

    # Organizations
    "Samsung", "삼성",
    "Hyundai", "현대",
    "LG", "엘지",
    "SK", "에스케이",
    "Lotte", "롯데",
    "Kakao", "카카오",
    "Naver", "네이버",
    "Coupang", "쿠팡",
    "KOSPI", "코스피",

    # Political
    "Blue House", "청와대",  # Former presidential residence
    "Yongsan", "용산",  # New presidential office
    "National Assembly", "국회",
    "Democratic Party", "민주당",
    "PPP", "국민의힘", "People Power Party",
    "prosecution", "검찰",
    "Constitutional Court", "헌법재판소",

    # Cultural
    "K-pop", "케이팝",
    "K-drama", "한드",
    "Hanbok", "한복",
    "Kimchi", "김치",
    "Soju", "소주",
    "PC bang", "PC방",
    "Noraebang", "노래방",
    "Hagwon", "학원",
    "Chaebol", "재벌",

    # Historical
    "Korean War", "한국전쟁", "6.25",
    "IMF crisis", "IMF 사태",
    "Gwangju Uprising", "광주민주화운동", "5.18",
    "Sewol", "세월호",
    "Itaewon crush", "이태원 참사",

    # Military
    "ROK Army", "육군",
    "conscription", "징병",
    "mandatory service", "병역",
    "THAAD", "사드",
    "USFK", "주한미군",
}

# Add all Korean-specific entities to the main set
ALL_KOREAN_ENTITIES.update(KOREAN_SPECIFIC_ENTITIES)
ALL_KOREAN_ENTITIES.update({e.lower() for e in KOREAN_SPECIFIC_ENTITIES})


def detect_korean_entities(text: str) -> List[str]:
    """
    Detect Korean entities mentioned in text.

    Args:
        text: Text to search for Korean entities

    Returns:
        List of Korean entities found in the text
    """
    if not text:
        return []

    found = []
    text_lower = text.lower()

    # Check each entity
    for entity in ALL_KOREAN_ENTITIES:
        # For Hangul, do exact match (case doesn't apply)
        if any('\uac00' <= c <= '\ud7a3' for c in entity):
            if entity in text:
                found.append(entity)
        # For English, do case-insensitive word boundary match
        else:
            entity_lower = entity.lower()
            # Simple contains check for now
            if entity_lower in text_lower:
                # Add the original (non-lowercased) version
                found.append(entity)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for entity in found:
        key = entity.lower() if entity.isascii() else entity
        if key not in seen:
            seen.add(key)
            unique.append(entity)

    return unique


def get_entity_mapping_context() -> str:
    """
    Get a formatted string of entity mappings for use in LLM prompts.

    Returns:
        Formatted string showing EN -> KR entity mappings
    """
    lines = ["Key cultural adaptations (English → Korean):"]

    # Select most important mappings for prompt context
    key_mappings = [
        ("Russia", "North Korea / 북한"),
        ("United States", "South Korea / 대한민국 / 한국"),
        ("JD Vance / US politician", "Lee Jun-seok / 이준석"),
        ("FBI/CIA", "NIS / 국정원"),
        ("Site R bunker", "Gwanaksan B-1 bunker / 관악산 벙커"),
        ("Kazan Airport", "Gimhae Airport / 김해공항"),
        ("Texas/California", "Seoul/Busan / 서울/부산"),
        ("New York Times", "Chosun Ilbo / 조선일보"),
        ("Google/Facebook", "Naver/Kakao / 네이버/카카오"),
    ]

    for en, kr in key_mappings:
        lines.append(f"  • {en} → {kr}")

    return "\n".join(lines)


# Export for easy access
__all__ = [
    "ENTITY_MAPPINGS",
    "ALL_KOREAN_ENTITIES",
    "KOREAN_SPECIFIC_ENTITIES",
    "detect_korean_entities",
    "get_entity_mapping_context",
]

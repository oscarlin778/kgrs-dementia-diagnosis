"""
Patient-Centric GraphRAG: 從 Neo4j 擷取個人化病患脈絡並轉換為自然語言描述。
"""

import os
import re
import functools
from typing import Optional

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

_driver = None


def _get_driver():
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver


def _extract_adni_id(subject_id: str) -> Optional[str]:
    """從各種格式的 subject_id 中抽取標準 ADNI ID (NNN_S_XXXX)。"""
    m = re.search(r"(\d{3}_S_\d{4})", subject_id)
    return m.group(1) if m else None


def _build_context_text(age: int, sex: str, education: int, clinical_dx: str) -> str:
    """將病患人口學屬性轉換為具有認知儲備脈絡的自然語言描述。"""
    dx_map = {
        "NC":  "正常認知（Normal Cognition, NC）",
        "MCI": "輕度認知障礙（Mild Cognitive Impairment, MCI）",
        "AD":  "阿茲海默症（Alzheimer's Disease, AD）",
    }
    dx_label = dx_map.get(clinical_dx, clinical_dx)

    # ── 年齡描述 ────────────────────────────────────────────────────
    unknown_age = (age == 0 or age is None)
    if unknown_age:
        age_desc = "年齡不詳"
    elif age < 65:
        age_desc = f"{age}歲（老年前期）"
    elif age < 75:
        age_desc = f"{age}歲（老年早期）"
    elif age < 85:
        age_desc = f"{age}歲（老年期）"
    else:
        age_desc = f"{age}歲（高齡）"

    # ── 性別描述 ────────────────────────────────────────────────────
    sex_desc = {"Male": "男性", "Female": "女性"}.get(sex, "性別不詳")

    # ── 認知儲備（CR）評估 ──────────────────────────────────────────
    unknown_edu = (education == 0 or education is None)
    if unknown_edu:
        cr_level = "認知儲備資料不完整"
        cr_note  = "缺乏教育年限資料，無法評估認知儲備，診斷需完全依賴影像分析。"
    elif education <= 8:
        cr_level = "認知儲備偏低（教育年限 ≤ 8 年）"
        cr_note  = (
            f"教育年限僅 {education} 年，認知儲備（Cognitive Reserve）相對不足，"
            "神經退化症狀可能較早出現且進展較快，影像病理損傷程度與症狀表現通常相符。"
        )
    elif education <= 12:
        cr_level = "中等認知儲備（教育年限 9–12 年）"
        cr_note  = (
            f"教育年限為 {education} 年，具備中等程度認知儲備，"
            "可能對早期神經退化有部分代償能力。"
        )
    elif education <= 16:
        cr_level = "良好認知儲備（教育年限 13–16 年）"
        cr_note  = (
            f"教育年限達 {education} 年，認知儲備充足，"
            "大腦具備較強的功能代償機制，症狀表現可能低估實際病理損傷程度；"
            "即使影像尚輕微，神經退化仍可能在較晚期才顯現於臨床行為。"
        )
    else:
        cr_level = "高度認知儲備（教育年限 > 16 年）"
        cr_note  = (
            f"教育年限達 {education} 年，認知儲備豐富，"
            "大腦具備高度的功能重組與代償能力。臨床症狀可能明顯低估影像所見的病理程度，"
            "診斷時需特別注意影像生物標記與認知測驗之間的乖離現象。"
        )

    # ── 診斷脈絡備注 ────────────────────────────────────────────────
    dx_note_map = {
        "NC":  "目前未有認知障礙，但影像監測有助於早期發現潛在退化。",
        "MCI": "已進入輕度認知障礙階段，需關注是否將進展為 AD；影像異常程度對預後評估至關重要。",
        "AD":  "確診阿茲海默症，影像分析可協助評估疾病進程與治療反應。",
    }
    dx_note = dx_note_map.get(clinical_dx, "")

    lines = [
        f"病患基本資料：{age_desc}、{sex_desc}，臨床診斷為 {dx_label}。",
        f"認知儲備評估：{cr_level}。{cr_note}",
    ]
    if dx_note:
        lines.append(f"診斷脈絡：{dx_note}")

    return "\n".join(lines)


@functools.lru_cache(maxsize=256)
def get_patient_graph_context(subject_id: str) -> str:
    """
    從 Neo4j 查詢 :Patient 節點並回傳自然語言脈絡描述。
    查詢不到時回傳「無此病患脈絡」的 fallback 字串。
    結果會被 LRU cache 快取，避免重複查詢。

    Args:
        subject_id: 任意格式，含 NNN_S_XXXX 的字串（如 '012_S_6760' 或 'sub-012_S_6760_MCI'）。

    Returns:
        自然語言描述字串，供注入 LLM System Prompt。
    """
    adni_id = _extract_adni_id(subject_id)
    if adni_id is None:
        return f"[病患脈絡] 無法從 '{subject_id}' 中解析出有效的 ADNI ID，脈絡資料缺失。"

    try:
        driver = _get_driver()
        with driver.session() as session:
            record = session.run(
                """
                MATCH (p:Patient {subject_id: $sid})
                RETURN p.age        AS age,
                       p.sex        AS sex,
                       p.education  AS education,
                       p.clinical_dx AS clinical_dx
                """,
                sid=adni_id,
            ).single()
    except Exception as exc:
        return f"[病患脈絡] Neo4j 查詢失敗（{exc}），報告將略過個人化脈絡。"

    if record is None:
        return (
            f"[病患脈絡] 知識圖譜中查無 subject_id='{adni_id}' 的病患資料，"
            "報告將依純影像分析結果生成，不含個人化背景資訊。"
        )

    return _build_context_text(
        age=record["age"] or 0,
        sex=record["sex"] or "Unknown",
        education=record["education"] or 0,
        clinical_dx=record["clinical_dx"] or "Unknown",
    )


def close_driver():
    """釋放 Neo4j driver（程式結束時呼叫）。"""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
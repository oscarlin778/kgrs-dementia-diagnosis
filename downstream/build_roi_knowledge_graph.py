"""
Build ROI Knowledge Graph for Neo4j
將 AAL116 ROI 知識、腦網路知識、疾病階段知識以及模型 attention difference
寫入 Neo4j Property Graph，作為 GraphRAG 的結構化知識底層。

Node types : ROI, BrainNetwork, DiseaseStage, Biomarker
Edge types  : BELONGS_TO, DISRUPTED_IN, ATTENTION_DIFF, ASSOCIATED_WITH

Usage:
    pip install neo4j
    python build_roi_knowledge_graph.py
"""

import json
import os

# ── Neo4j 連線設定（依環境修改）──────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "neo4j://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ═══════════════════════════════════════════════════════════════════
# 1. ROI 知識對照表（116 個 AAL 腦區）
# ═══════════════════════════════════════════════════════════════════
ROI_KNOWLEDGE = [
    # ── 額葉 Frontal ──────────────────────────────────────────────
    {"index": 0,  "name": "Precentral_L",       "hemisphere": "L", "lobe": "Frontal",
     "function": "Primary motor cortex; voluntary movement execution",
     "ad_relevance": "low",
     "notes": "Relatively preserved in early AD; affected in late-stage motor decline"},
    {"index": 1,  "name": "Precentral_R",       "hemisphere": "R", "lobe": "Frontal",
     "function": "Primary motor cortex; voluntary movement execution",
     "ad_relevance": "low",
     "notes": "Relatively preserved in early AD; affected in late-stage motor decline"},
    {"index": 2,  "name": "Frontal_Sup_L",      "hemisphere": "L", "lobe": "Frontal",
     "function": "Superior frontal gyrus; executive function, working memory, planning",
     "ad_relevance": "medium",
     "notes": "Shows reduced FC with posterior regions in MCI; part of cognitive control network"},
    {"index": 3,  "name": "Frontal_Sup_R",      "hemisphere": "R", "lobe": "Frontal",
     "function": "Superior frontal gyrus; executive function, working memory, planning",
     "ad_relevance": "medium",
     "notes": "Shows reduced FC with posterior regions in MCI; part of cognitive control network"},
    {"index": 4,  "name": "Frontal_Sup_Orb_L",  "hemisphere": "L", "lobe": "Frontal",
     "function": "Superior frontal gyrus orbital part; emotion regulation, reward processing",
     "ad_relevance": "medium",
     "notes": "OFC connected to limbic system; shows tau pathology in mid-stage AD"},
    {"index": 5,  "name": "Frontal_Sup_Orb_R",  "hemisphere": "R", "lobe": "Frontal",
     "function": "Superior frontal gyrus orbital part; emotion regulation, reward processing",
     "ad_relevance": "medium",
     "notes": "OFC connected to limbic system; shows tau pathology in mid-stage AD"},
    {"index": 6,  "name": "Frontal_Mid_L",       "hemisphere": "L", "lobe": "Frontal",
     "function": "Middle frontal gyrus (DLPFC); executive function, working memory, top-down attention",
     "ad_relevance": "high",
     "notes": "FPN hub; FC disruption with parietal regions is early MCI marker; left DLPFC critical for episodic memory encoding"},
    {"index": 7,  "name": "Frontal_Mid_R",       "hemisphere": "R", "lobe": "Frontal",
     "function": "Middle frontal gyrus (DLPFC); executive function, working memory, top-down attention",
     "ad_relevance": "high",
     "notes": "FPN hub; right DLPFC involved in attention control; FC changes correlate with cognitive scores"},
    {"index": 8,  "name": "Frontal_Mid_Orb_L",   "hemisphere": "L", "lobe": "Frontal",
     "function": "Middle frontal gyrus orbital part; decision-making, impulse control",
     "ad_relevance": "medium",
     "notes": "Part of ventromedial prefrontal circuit; affected in behavioral variant frontotemporal dementia"},
    {"index": 9,  "name": "Frontal_Mid_Orb_R",   "hemisphere": "R", "lobe": "Frontal",
     "function": "Middle frontal gyrus orbital part; decision-making, impulse control",
     "ad_relevance": "medium",
     "notes": "Part of ventromedial prefrontal circuit; affected in behavioral variant frontotemporal dementia"},
    {"index": 10, "name": "Frontal_Inf_Oper_L",  "hemisphere": "L", "lobe": "Frontal",
     "function": "Inferior frontal gyrus opercular part (Broca's area); speech production, phonological processing",
     "ad_relevance": "low",
     "notes": "Language production; affected in logopenic variant PPA; VAN node"},
    {"index": 11, "name": "Frontal_Inf_Oper_R",  "hemisphere": "R", "lobe": "Frontal",
     "function": "Inferior frontal gyrus opercular part; prosody, pragmatic language",
     "ad_relevance": "low",
     "notes": "Right hemisphere language prosody; VAN node"},
    {"index": 12, "name": "Frontal_Inf_Tri_L",   "hemisphere": "L", "lobe": "Frontal",
     "function": "Inferior frontal gyrus triangular part (Broca's area); semantic processing, language comprehension",
     "ad_relevance": "low",
     "notes": "Core language area; semantic memory access"},
    {"index": 13, "name": "Frontal_Inf_Tri_R",   "hemisphere": "R", "lobe": "Frontal",
     "function": "Inferior frontal gyrus triangular part; pragmatic comprehension, inference",
     "ad_relevance": "low",
     "notes": "Discourse-level language processing"},
    {"index": 14, "name": "Frontal_Inf_Orb_L",   "hemisphere": "L", "lobe": "Frontal",
     "function": "Inferior frontal gyrus orbital part; semantic memory, response inhibition",
     "ad_relevance": "medium",
     "notes": "VAN node; connected to limbic-OFC circuit; affected in semantic dementia"},
    {"index": 15, "name": "Frontal_Inf_Orb_R",   "hemisphere": "R", "lobe": "Frontal",
     "function": "Inferior frontal gyrus orbital part; emotional semantic processing",
     "ad_relevance": "medium",
     "notes": "VAN node; reward-based inhibitory control"},
    {"index": 16, "name": "Rolandic_Oper_L",     "hemisphere": "L", "lobe": "Frontal",
     "function": "Rolandic operculum; speech articulation, swallowing coordination",
     "ad_relevance": "low",
     "notes": "Motor-sensory integration for orofacial movements"},
    {"index": 17, "name": "Rolandic_Oper_R",     "hemisphere": "R", "lobe": "Frontal",
     "function": "Rolandic operculum; speech articulation, swallowing coordination",
     "ad_relevance": "low",
     "notes": "Motor-sensory integration for orofacial movements"},
    {"index": 18, "name": "Supp_Motor_Area_L",   "hemisphere": "L", "lobe": "Frontal",
     "function": "Supplementary motor area; motor planning, sequential movement, bilateral coordination",
     "ad_relevance": "low",
     "notes": "Pre-movement planning; connected to basal ganglia; SMN-adjacent"},
    {"index": 19, "name": "Supp_Motor_Area_R",   "hemisphere": "R", "lobe": "Frontal",
     "function": "Supplementary motor area; motor planning, sequential movement, bilateral coordination",
     "ad_relevance": "low",
     "notes": "Pre-movement planning; connected to basal ganglia; SMN-adjacent"},
    {"index": 20, "name": "Olfactory_L",         "hemisphere": "L", "lobe": "Frontal",
     "function": "Olfactory cortex / piriform cortex; primary olfactory processing, emotional memory",
     "ad_relevance": "high",
     "notes": "CRITICAL EARLY BIOMARKER: Braak Stage I-II tau accumulation begins here; olfactory deficits are among earliest AD symptoms (3-5 years before diagnosis); connected to entorhinal cortex and hippocampus via perforant path"},
    {"index": 21, "name": "Olfactory_R",         "hemisphere": "R", "lobe": "Frontal",
     "function": "Olfactory cortex / piriform cortex; primary olfactory processing, emotional memory",
     "ad_relevance": "high",
     "notes": "CRITICAL EARLY BIOMARKER: Braak Stage I-II tau accumulation begins here; olfactory deficits are among earliest AD symptoms; model shows this as top AD-discriminative ROI in MCI vs AD task"},
    {"index": 22, "name": "Frontal_Sup_Med_L",   "hemisphere": "L", "lobe": "Frontal",
     "function": "Medial superior frontal gyrus; self-referential processing, social cognition, mentalizing",
     "ad_relevance": "high",
     "notes": "DMN node; default mode hub; shows hypometabolism in early AD; involved in autobiographical memory retrieval"},
    {"index": 23, "name": "Frontal_Sup_Med_R",   "hemisphere": "R", "lobe": "Frontal",
     "function": "Medial superior frontal gyrus; self-referential processing, social cognition, mentalizing",
     "ad_relevance": "high",
     "notes": "DMN node; default mode hub; shows hypometabolism in early AD"},
    {"index": 24, "name": "Frontal_Med_Orb_L",   "hemisphere": "L", "lobe": "Frontal",
     "function": "Medial orbitofrontal cortex; emotion-cognition integration, moral reasoning",
     "ad_relevance": "high",
     "notes": "DMN node; vmPFC; critical for emotion regulation and social decision-making; Aβ deposition observed in preclinical AD"},
    {"index": 25, "name": "Frontal_Med_Orb_R",   "hemisphere": "R", "lobe": "Frontal",
     "function": "Medial orbitofrontal cortex; emotion-cognition integration, moral reasoning",
     "ad_relevance": "high",
     "notes": "DMN node; vmPFC; connected to amygdala and hippocampus; affected in behavioral symptoms of AD"},
    {"index": 26, "name": "Rectus_L",            "hemisphere": "L", "lobe": "Frontal",
     "function": "Gyrus rectus; emotional decision-making, reward valuation, social behavior",
     "ad_relevance": "medium",
     "notes": "Shows volume loss in AD; located in subgenual OFC; connected to limbic circuits; compensatory FC increases reported in early AD"},
    {"index": 27, "name": "Rectus_R",            "hemisphere": "R", "lobe": "Frontal",
     "function": "Gyrus rectus; emotional decision-making, reward valuation, social behavior",
     "ad_relevance": "medium",
     "notes": "Shows volume loss in AD; right hemisphere reward circuit; model shows high importance in MCI vs AD"},

    # ── 島葉 / 扣帶迴 Insula / Cingulate ─────────────────────────
    {"index": 28, "name": "Insula_L",            "hemisphere": "L", "lobe": "Insula",
     "function": "Insula; interoceptive awareness, emotional salience, pain processing, autonomic regulation",
     "ad_relevance": "high",
     "notes": "SN hub; anterior insula is critical for salience detection; FC disruption is early MCI marker; connected to ACC and amygdala; model shows NC>MCI in attention (SN=-0.32 in NC vs MCI)"},
    {"index": 29, "name": "Insula_R",            "hemisphere": "R", "lobe": "Insula",
     "function": "Insula; interoceptive awareness, emotional salience, pain processing",
     "ad_relevance": "high",
     "notes": "SN hub; right insula particularly important for body awareness and autonomic control; shows early FC changes in MCI"},
    {"index": 30, "name": "Cingulum_Ant_L",      "hemisphere": "L", "lobe": "Cingulate",
     "function": "Anterior cingulate cortex; conflict monitoring, error detection, emotional regulation, attention",
     "ad_relevance": "high",
     "notes": "SN node; neurofibrillary tangles appear in Braak Stage IV; key node for cognitive control; FC with insula disrupted in MCI"},
    {"index": 31, "name": "Cingulum_Ant_R",      "hemisphere": "R", "lobe": "Cingulate",
     "function": "Anterior cingulate cortex; conflict monitoring, pain affect",
     "ad_relevance": "high",
     "notes": "SN node; dorsal ACC for cognitive control; subgenual ACC for emotional processing"},
    {"index": 32, "name": "Cingulum_Mid_L",      "hemisphere": "L", "lobe": "Cingulate",
     "function": "Middle cingulate cortex; motor-limbic integration, pain processing, attention to action",
     "ad_relevance": "medium",
     "notes": "SN node; connects motor and limbic systems; cingulum bundle integrity (DTI) is AD biomarker"},
    {"index": 33, "name": "Cingulum_Mid_R",      "hemisphere": "R", "lobe": "Cingulate",
     "function": "Middle cingulate cortex; motor-limbic integration, skeletomotor body orientation",
     "ad_relevance": "medium",
     "notes": "SN node; cingulum bundle white matter tract connects hippocampus and PCC"},
    {"index": 34, "name": "Cingulum_Post_L",     "hemisphere": "L", "lobe": "Cingulate",
     "function": "Posterior cingulate cortex; default mode, self-referential processing, autobiographical memory, visuospatial orientation",
     "ad_relevance": "high",
     "notes": "DMN hub; CRITICAL: one of the earliest hypometabolism sites in preclinical AD (FDG-PET); Aβ accumulates here first; FC reduction with hippocampus is early MCI marker; model shows NC>AD attention (NC attention preserved here)"},
    {"index": 35, "name": "Cingulum_Post_R",     "hemisphere": "R", "lobe": "Cingulate",
     "function": "Posterior cingulate cortex; default mode, episodic memory retrieval, spatial navigation",
     "ad_relevance": "high",
     "notes": "DMN hub; PCC/precuneus complex is primary site of Aβ deposition in preclinical AD; retrograde memory"},

    # ── 邊緣系統 Limbic ───────────────────────────────────────────
    {"index": 36, "name": "Hippocampus_L",       "hemisphere": "L", "lobe": "Limbic",
     "function": "Hippocampus; episodic memory encoding and consolidation, spatial navigation (cognitive map)",
     "ad_relevance": "high",
     "notes": "PRIMARY AD BIOMARKER: Braak Stage III-IV tau accumulation; left hippocampal volume loss is primary MRI diagnostic criterion; entorhinal-hippocampal FC disruption earliest FC change; LN model shows NC>>MCI in attention (LN=-0.33)"},
    {"index": 37, "name": "Hippocampus_R",       "hemisphere": "R", "lobe": "Limbic",
     "function": "Hippocampus; spatial memory, contextual fear conditioning, pattern separation",
     "ad_relevance": "high",
     "notes": "PRIMARY AD BIOMARKER: right hippocampal atrophy rate >3%/year in AD vs <1%/year in NC; bilateral hippocampal volume correlates with MMSE; model shows LN disruption in early MCI"},
    {"index": 38, "name": "ParaHippocampal_L",   "hemisphere": "L", "lobe": "Limbic",
     "function": "Parahippocampal gyrus; memory encoding support, scene recognition, spatial context",
     "ad_relevance": "high",
     "notes": "Contains entorhinal cortex (Braak Stage I-II); gateway to hippocampus; first site of NFT pathology; perirhinal cortex mediates semantic memory; early atrophy in MCI"},
    {"index": 39, "name": "ParaHippocampal_R",   "hemisphere": "R", "lobe": "Limbic",
     "function": "Parahippocampal gyrus; scene recognition, spatial navigation, allocentric memory",
     "ad_relevance": "high",
     "notes": "Contains entorhinal cortex; right PHC critical for spatial scene processing; topographical disorientation in early AD"},
    {"index": 40, "name": "Amygdala_L",          "hemisphere": "L", "lobe": "Limbic",
     "function": "Amygdala; emotional processing, fear conditioning, social emotion recognition",
     "ad_relevance": "high",
     "notes": "Early atrophy in AD along with hippocampus; connected to hippocampus via fornix; emotional memory consolidation; apathy and anxiety symptoms in MCI/AD linked to amygdala degeneration"},
    {"index": 41, "name": "Amygdala_R",          "hemisphere": "R", "lobe": "Limbic",
     "function": "Amygdala; threat detection, emotional arousal, social behavior",
     "ad_relevance": "high",
     "notes": "Right amygdala shows early tau pathology; volume loss correlates with neuropsychiatric symptoms in AD; model shows AD>MCI in LN attention"},

    # ── 枕葉 / 視覺 Visual Cortex ─────────────────────────────────
    {"index": 42, "name": "Calcarine_L",         "hemisphere": "L", "lobe": "Occipital",
     "function": "Calcarine sulcus / primary visual cortex (V1); basic visual processing, contrast detection",
     "ad_relevance": "low",
     "notes": "VN; generally preserved until late AD; affected in posterior cortical atrophy (PCA) variant; Aβ deposits late"},
    {"index": 43, "name": "Calcarine_R",         "hemisphere": "R", "lobe": "Occipital",
     "function": "Calcarine sulcus / primary visual cortex (V1); basic visual processing",
     "ad_relevance": "low",
     "notes": "VN; primary visual processing; preserved in typical AD"},
    {"index": 44, "name": "Cuneus_L",            "hemisphere": "L", "lobe": "Occipital",
     "function": "Cuneus; secondary visual cortex, visuospatial processing, visual attention",
     "ad_relevance": "medium",
     "notes": "VN; involved in PCA variant of AD; visuospatial deficits; model shows MCI has highest VN attention (compensatory hyperconnectivity)"},
    {"index": 45, "name": "Cuneus_R",            "hemisphere": "R", "lobe": "Occipital",
     "function": "Cuneus; visuospatial attention, depth perception",
     "ad_relevance": "medium",
     "notes": "VN; model shows Cuneus_R as top ROI in MCI vs AD differential"},
    {"index": 46, "name": "Lingual_L",           "hemisphere": "L", "lobe": "Occipital",
     "function": "Lingual gyrus; visual word form processing, color perception, dream imagery",
     "ad_relevance": "medium",
     "notes": "VN; visual reading circuit; connected to memory systems; shows FC changes in AD"},
    {"index": 47, "name": "Lingual_R",           "hemisphere": "R", "lobe": "Occipital",
     "function": "Lingual gyrus; face/scene processing, topographic memory",
     "ad_relevance": "medium",
     "notes": "VN; model shows Lingual_R as top NC-dominant ROI in NC vs AD (NC attention higher); important in DMN-visual coupling"},
    {"index": 48, "name": "Occipital_Sup_L",     "hemisphere": "L", "lobe": "Occipital",
     "function": "Superior occipital gyrus; higher visual processing, spatial attention, dorsal visual stream",
     "ad_relevance": "medium",
     "notes": "VN; dorsal stream for visuomotor coordination; involved in PCA variant"},
    {"index": 49, "name": "Occipital_Sup_R",     "hemisphere": "R", "lobe": "Occipital",
     "function": "Superior occipital gyrus; spatial perception, visual guidance of movement",
     "ad_relevance": "medium",
     "notes": "VN; right hemisphere spatial processing; FC changes with FPN in MCI"},
    {"index": 50, "name": "Occipital_Mid_L",     "hemisphere": "L", "lobe": "Occipital",
     "function": "Middle occipital gyrus; motion detection, object recognition, extrastriate cortex",
     "ad_relevance": "medium",
     "notes": "VN; V5/MT area for motion processing; shows FC hyperconnectivity in early MCI"},
    {"index": 51, "name": "Occipital_Mid_R",     "hemisphere": "R", "lobe": "Occipital",
     "function": "Middle occipital gyrus; motion detection, visual scene analysis",
     "ad_relevance": "medium",
     "notes": "VN; right occipital shows strong FC changes in NC vs MCI comparison"},
    {"index": 52, "name": "Occipital_Inf_L",     "hemisphere": "L", "lobe": "Occipital",
     "function": "Inferior occipital gyrus; ventral visual stream, object/face recognition",
     "ad_relevance": "medium",
     "notes": "VN; ventral stream entry to fusiform; model shows this as top differential ROI in NC vs AD"},
    {"index": 53, "name": "Occipital_Inf_R",     "hemisphere": "R", "lobe": "Occipital",
     "function": "Inferior occipital gyrus; ventral visual stream, face processing entry",
     "ad_relevance": "medium",
     "notes": "VN; top differential ROI in NC vs AD; right occipital shows consistent changes"},
    {"index": 54, "name": "Fusiform_L",          "hemisphere": "L", "lobe": "Temporal",
     "function": "Fusiform gyrus; face recognition (FFA), word recognition (VWFA), object category processing",
     "ad_relevance": "high",
     "notes": "Fusiform face area; atrophy correlates with face recognition deficits; semantic memory access; shows volume reduction in typical AD"},
    {"index": 55, "name": "Fusiform_R",          "hemisphere": "R", "lobe": "Temporal",
     "function": "Fusiform gyrus; face recognition, visual object expertise",
     "ad_relevance": "high",
     "notes": "Right fusiform predominant for holistic face processing; degree centrality reduction in AD"},

    # ── 頂葉 Parietal ─────────────────────────────────────────────
    {"index": 56, "name": "Postcentral_L",       "hemisphere": "L", "lobe": "Parietal",
     "function": "Postcentral gyrus / primary somatosensory cortex (S1); tactile and proprioceptive processing",
     "ad_relevance": "low",
     "notes": "SMN; primary sensory cortex; relatively preserved in typical AD"},
    {"index": 57, "name": "Postcentral_R",       "hemisphere": "R", "lobe": "Parietal",
     "function": "Postcentral gyrus / primary somatosensory cortex (S1); touch, pain, body position",
     "ad_relevance": "low",
     "notes": "SMN; sensory processing preserved until late AD"},
    {"index": 58, "name": "Parietal_Sup_L",      "hemisphere": "L", "lobe": "Parietal",
     "function": "Superior parietal lobule; visuomotor integration, spatial attention, tool use",
     "ad_relevance": "high",
     "notes": "FPN node; dorsal attention network; apraxia in AD linked to SPL; FC disruption with DLPFC in MCI"},
    {"index": 59, "name": "Parietal_Sup_R",      "hemisphere": "R", "lobe": "Parietal",
     "function": "Superior parietal lobule; spatial attention (right-dominant), attention shifting",
     "ad_relevance": "high",
     "notes": "FPN node; right SPL critical for visuospatial attention; hemineglect in advanced AD"},
    {"index": 60, "name": "Parietal_Inf_L",      "hemisphere": "L", "lobe": "Parietal",
     "function": "Inferior parietal lobule; language processing, spatial reasoning, semantic integration",
     "ad_relevance": "high",
     "notes": "FPN node; includes supramarginal and angular gyri area; language-dominant left IPL; significant atrophy in AD"},
    {"index": 61, "name": "Parietal_Inf_R",      "hemisphere": "R", "lobe": "Parietal",
     "function": "Inferior parietal lobule; spatial processing, gesture recognition, number cognition",
     "ad_relevance": "high",
     "notes": "FPN node; right IPL for visuospatial and spatial working memory; acalculia in AD"},
    {"index": 62, "name": "SupraMarginal_L",     "hemisphere": "L", "lobe": "Parietal",
     "function": "Supramarginal gyrus; phonological processing, working memory, empathy",
     "ad_relevance": "medium",
     "notes": "Part of phonological loop; language decoding; affected in logopenic AD variant"},
    {"index": 63, "name": "SupraMarginal_R",     "hemisphere": "R", "lobe": "Parietal",
     "function": "Supramarginal gyrus; emotional empathy, social cognition",
     "ad_relevance": "medium",
     "notes": "Right SMG for emotional prosody interpretation"},
    {"index": 64, "name": "Angular_L",           "hemisphere": "L", "lobe": "Parietal",
     "function": "Angular gyrus; semantic integration, reading comprehension, numerical cognition, attention",
     "ad_relevance": "high",
     "notes": "DMN node; multimodal semantic hub; left AG critical for reading and semantic memory retrieval; highly connected to PCC in DMN; FC reduction is key AD marker"},
    {"index": 65, "name": "Angular_R",           "hemisphere": "R", "lobe": "Parietal",
     "function": "Angular gyrus; spatial attention, semantic memory, self-referential thought",
     "ad_relevance": "high",
     "notes": "DMN node; right AG important for spatial reasoning; model shows Angular_R as top NC-dominant ROI in NC vs MCI"},
    {"index": 66, "name": "Precuneus_L",         "hemisphere": "L", "lobe": "Parietal",
     "function": "Precuneus; visuospatial imagery, episodic memory retrieval, self-consciousness, mental rotation",
     "ad_relevance": "high",
     "notes": "DMN hub; EARLIEST amyloid deposition site in preclinical AD; significant hypometabolism; FC disruption with PCC and hippocampus is most reliable early AD marker; model shows NC>AD in attention (Precuneus NC-dominant)"},
    {"index": 67, "name": "Precuneus_R",         "hemisphere": "R", "lobe": "Parietal",
     "function": "Precuneus; visuospatial processing, self-reflection, consciousness",
     "ad_relevance": "high",
     "notes": "DMN hub; right Precuneus shows Aβ accumulation in ADNI preclinical cohort; model shows Precuneus_R as AD-dominant ROI in MCI vs AD"},
    {"index": 68, "name": "Paracentral_Lob_L",   "hemisphere": "L", "lobe": "Parietal",
     "function": "Paracentral lobule; lower limb motor and sensory cortex, micturition control",
     "ad_relevance": "low",
     "notes": "SMN; motor and sensory representation of lower extremities; gait control"},
    {"index": 69, "name": "Paracentral_Lob_R",   "hemisphere": "R", "lobe": "Parietal",
     "function": "Paracentral lobule; lower limb motor and sensory cortex",
     "ad_relevance": "low",
     "notes": "SMN; gait and balance; falls risk in late AD"},

    # ── 皮質下 Subcortical / BGN ──────────────────────────────────
    {"index": 70, "name": "Caudate_L",           "hemisphere": "L", "lobe": "Subcortical",
     "function": "Caudate nucleus; procedural learning, executive function, reward-based learning",
     "ad_relevance": "medium",
     "notes": "BGN; connected to DLPFC via cortico-striatal loop; dopaminergic; FC with prefrontal shows changes in MCI"},
    {"index": 71, "name": "Caudate_R",           "hemisphere": "R", "lobe": "Subcortical",
     "function": "Caudate nucleus; spatial working memory, habit formation",
     "ad_relevance": "medium",
     "notes": "BGN; right caudate for spatial learning; volume reduced in late AD"},
    {"index": 72, "name": "Putamen_L",           "hemisphere": "L", "lobe": "Subcortical",
     "function": "Putamen; motor sequence learning, procedural memory, reward processing",
     "ad_relevance": "low",
     "notes": "BGN; dopaminergic; motor circuit hub; relatively preserved in early AD; involved in Lewy body dementia differential"},
    {"index": 73, "name": "Putamen_R",           "hemisphere": "R", "lobe": "Subcortical",
     "function": "Putamen; motor learning, habitual behavior",
     "ad_relevance": "low",
     "notes": "BGN; dopamine release for reward; model shows Putamen_R as AD-important in MCI vs AD"},
    {"index": 74, "name": "Pallidum_L",          "hemisphere": "L", "lobe": "Subcortical",
     "function": "Globus pallidus; motor control modulation, basal ganglia output",
     "ad_relevance": "low",
     "notes": "BGN; relay station for motor control; inhibitory GABAergic output to thalamus"},
    {"index": 75, "name": "Pallidum_R",          "hemisphere": "R", "lobe": "Subcortical",
     "function": "Globus pallidus; motor control, movement suppression",
     "ad_relevance": "low",
     "notes": "BGN; motor control circuit; affected in extrapyramidal symptoms"},
    {"index": 76, "name": "Thalamus_L",          "hemisphere": "L", "lobe": "Subcortical",
     "function": "Thalamus; sensory relay, consciousness regulation, corticothalamic gating",
     "ad_relevance": "medium",
     "notes": "BGN; thalamic atrophy correlates with global cognitive decline; mediodorsal nucleus connects to prefrontal; pulvinar connects to posterior cortex; FC disruption widespread in AD"},
    {"index": 77, "name": "Thalamus_R",          "hemisphere": "R", "lobe": "Subcortical",
     "function": "Thalamus; sensory integration, sleep regulation, attention gating",
     "ad_relevance": "medium",
     "notes": "BGN; right thalamus for spatial attention; sleep dysregulation in MCI/AD linked to thalamic changes"},

    # ── 顳葉 Temporal ─────────────────────────────────────────────
    {"index": 78, "name": "Heschl_L",            "hemisphere": "L", "lobe": "Temporal",
     "function": "Heschl's gyrus / primary auditory cortex (A1); basic auditory processing, pitch detection",
     "ad_relevance": "low",
     "notes": "Primary auditory; preserved in typical AD; affected in some dementia variants"},
    {"index": 79, "name": "Heschl_R",            "hemisphere": "R", "lobe": "Temporal",
     "function": "Heschl's gyrus / primary auditory cortex; music perception, tonal processing",
     "ad_relevance": "low",
     "notes": "Primary auditory; right dominant for music and prosody"},
    {"index": 80, "name": "Temporal_Sup_L",      "hemisphere": "L", "lobe": "Temporal",
     "function": "Superior temporal gyrus; auditory association, speech comprehension (Wernicke's area)",
     "ad_relevance": "high",
     "notes": "Wernicke's area; language comprehension; atrophy in typical and logopenic variant AD; STS for biological motion"},
    {"index": 81, "name": "Temporal_Sup_R",      "hemisphere": "R", "lobe": "Temporal",
     "function": "Superior temporal gyrus; prosody, social cognition, biological motion",
     "ad_relevance": "high",
     "notes": "Right STG for social cognition and theory of mind; atrophy correlates with social symptoms in AD"},
    {"index": 82, "name": "Temporal_Pole_Sup_L", "hemisphere": "L", "lobe": "Temporal",
     "function": "Temporal pole superior; semantic memory, conceptual knowledge, social emotion",
     "ad_relevance": "high",
     "notes": "Semantic memory hub; early atrophy in semantic dementia and temporal variant AD; connected to amygdala and OFC"},
    {"index": 83, "name": "Temporal_Pole_Sup_R", "hemisphere": "R", "lobe": "Temporal",
     "function": "Temporal pole superior; social cognition, autobiographical memory, emotional semantics",
     "ad_relevance": "high",
     "notes": "Right temporal pole for person-specific semantic knowledge and social memory"},
    {"index": 84, "name": "Temporal_Mid_L",      "hemisphere": "L", "lobe": "Temporal",
     "function": "Middle temporal gyrus; semantic memory, language comprehension, tool knowledge",
     "ad_relevance": "high",
     "notes": "Core semantic memory region; significant atrophy in typical AD; FC changes with frontal semantic regions; model shows Temporal_Mid as top AD differential ROI"},
    {"index": 85, "name": "Temporal_Mid_R",      "hemisphere": "R", "lobe": "Temporal",
     "function": "Middle temporal gyrus; semantic memory, narrative comprehension, biological motion",
     "ad_relevance": "high",
     "notes": "Temporal_Mid_R is top AD-discriminative ROI in NC vs AD; semantic processing disruption; widely reported in AD neuroimaging"},
    {"index": 86, "name": "Temporal_Pole_Mid_L", "hemisphere": "L", "lobe": "Temporal",
     "function": "Temporal pole middle; conceptual knowledge, cross-modal semantic binding",
     "ad_relevance": "high",
     "notes": "Semantic hub; early atrophy in semantic dementia; connects visual and verbal semantic systems"},
    {"index": 87, "name": "Temporal_Pole_Mid_R", "hemisphere": "R", "lobe": "Temporal",
     "function": "Temporal pole middle; social semantic memory, face-name associations",
     "ad_relevance": "high",
     "notes": "Face-name binding; famous person recognition deficits in AD"},
    {"index": 88, "name": "Temporal_Inf_L",      "hemisphere": "L", "lobe": "Temporal",
     "function": "Inferior temporal gyrus; ventral visual stream, object recognition, semantic categorization",
     "ad_relevance": "high",
     "notes": "Category-selective object recognition; visual semantic processing; atrophy in typical AD"},
    {"index": 89, "name": "Temporal_Inf_R",      "hemisphere": "R", "lobe": "Temporal",
     "function": "Inferior temporal gyrus; face and object recognition, visual expertise",
     "ad_relevance": "high",
     "notes": "Right IT for visual object categorization; connected to fusiform face area"},

    # ── 小腦 Cerebellum ───────────────────────────────────────────
    {"index": 90, "name": "Cerebelum_Crus1_L",   "hemisphere": "L", "lobe": "Cerebellum",
     "function": "Cerebellar Crus I; cognitive processing, language, connected to prefrontal via dentato-thalamo-cortical tract",
     "ad_relevance": "medium",
     "notes": "CereN; cognitive cerebellum; FC with DMN and FPN; FC changes observed in AD"},
    {"index": 91, "name": "Cerebelum_Crus1_R",   "hemisphere": "R", "lobe": "Cerebellum",
     "function": "Cerebellar Crus I; cognitive processing, executive function support",
     "ad_relevance": "medium",
     "notes": "CereN; connected to contralateral prefrontal; cognitive cerebellar circuit"},
    {"index": 92, "name": "Cerebelum_Crus2_L",   "hemisphere": "L", "lobe": "Cerebellum",
     "function": "Cerebellar Crus II; cognitive and social processing, DMN-cerebellar connectivity",
     "ad_relevance": "medium",
     "notes": "CereN; Crus I/II form cognitive cerebellum; show FC with DMN regions; changes in AD"},
    {"index": 93, "name": "Cerebelum_Crus2_R",   "hemisphere": "R", "lobe": "Cerebellum",
     "function": "Cerebellar Crus II; higher cognitive function support",
     "ad_relevance": "medium",
     "notes": "CereN; right cerebellar Crus II connected to left prefrontal (crossed laterality)"},
    {"index": 94, "name": "Cerebelum_3_L",        "hemisphere": "L", "lobe": "Cerebellum",
     "function": "Cerebellar lobule III; sensorimotor coordination",
     "ad_relevance": "low",
     "notes": "CereN; motor cerebellum; balance and gait"},
    {"index": 95, "name": "Cerebelum_3_R",        "hemisphere": "R", "lobe": "Cerebellum",
     "function": "Cerebellar lobule III; sensorimotor coordination",
     "ad_relevance": "low",
     "notes": "CereN; motor coordination"},
    {"index": 96, "name": "Cerebelum_4_5_L",      "hemisphere": "L", "lobe": "Cerebellum",
     "function": "Cerebellar lobules IV-V; motor control, limb coordination",
     "ad_relevance": "low",
     "notes": "CereN; anterior cerebellar lobe; motor execution"},
    {"index": 97, "name": "Cerebelum_4_5_R",      "hemisphere": "R", "lobe": "Cerebellum",
     "function": "Cerebellar lobules IV-V; limb motor coordination",
     "ad_relevance": "low",
     "notes": "CereN; motor cerebellar lobule"},
    {"index": 98, "name": "Cerebelum_6_L",        "hemisphere": "L", "lobe": "Cerebellum",
     "function": "Cerebellar lobule VI; sensorimotor and cognitive processing transition zone",
     "ad_relevance": "low",
     "notes": "CereN; lobule VI is interface between motor and cognitive cerebellum"},
    {"index": 99, "name": "Cerebelum_6_R",        "hemisphere": "R", "lobe": "Cerebellum",
     "function": "Cerebellar lobule VI; sensorimotor integration",
     "ad_relevance": "low",
     "notes": "CereN; sensorimotor integration"},
    {"index": 100, "name": "Cerebelum_7b_L",      "hemisphere": "L", "lobe": "Cerebellum",
     "function": "Cerebellar lobule VIIb; multisensory integration",
     "ad_relevance": "low",
     "notes": "CereN; posterior cerebellar lobe"},
    {"index": 101, "name": "Cerebelum_7b_R",      "hemisphere": "R", "lobe": "Cerebellum",
     "function": "Cerebellar lobule VIIb; multisensory integration",
     "ad_relevance": "low",
     "notes": "CereN; posterior cerebellar lobe"},
    {"index": 102, "name": "Cerebelum_8_L",       "hemisphere": "L", "lobe": "Cerebellum",
     "function": "Cerebellar lobule VIII; sensorimotor processing, balance",
     "ad_relevance": "low",
     "notes": "CereN; connected to vestibular nuclei"},
    {"index": 103, "name": "Cerebelum_8_R",       "hemisphere": "R", "lobe": "Cerebellum",
     "function": "Cerebellar lobule VIII; balance, locomotion",
     "ad_relevance": "low",
     "notes": "CereN; fall risk in AD relates to cerebellar dysfunction"},
    {"index": 104, "name": "Cerebelum_9_L",       "hemisphere": "L", "lobe": "Cerebellum",
     "function": "Cerebellar lobule IX; vestibular processing, oculomotor control",
     "ad_relevance": "low",
     "notes": "CereN; eye movement coordination"},
    {"index": 105, "name": "Cerebelum_9_R",       "hemisphere": "R", "lobe": "Cerebellum",
     "function": "Cerebellar lobule IX; vestibular and autonomic processing",
     "ad_relevance": "low",
     "notes": "CereN; cerebellar-limbic connection"},
    {"index": 106, "name": "Cerebelum_10_L",      "hemisphere": "L", "lobe": "Cerebellum",
     "function": "Cerebellar lobule X (flocculus); vestibular processing, balance reflex",
     "ad_relevance": "medium",
     "notes": "CereN; model shows Cerebellum_10 as top blue ROI in MCI vs AD (MCI has higher attention)"},
    {"index": 107, "name": "Cerebelum_10_R",      "hemisphere": "R", "lobe": "Cerebellum",
     "function": "Cerebellar lobule X (flocculus); vestibular processing, gaze stabilization",
     "ad_relevance": "medium",
     "notes": "CereN; oculomotor and balance control"},
    {"index": 108, "name": "Vermis_1_2",          "hemisphere": "M", "lobe": "Cerebellum",
     "function": "Cerebellar vermis lobules I-II; motor coordination midline",
     "ad_relevance": "low",
     "notes": "CereN; vermis for axial motor control"},
    {"index": 109, "name": "Vermis_3",            "hemisphere": "M", "lobe": "Cerebellum",
     "function": "Cerebellar vermis lobule III; balance coordination",
     "ad_relevance": "low",
     "notes": "CereN; midline cerebellar"},
    {"index": 110, "name": "Vermis_4_5",          "hemisphere": "M", "lobe": "Cerebellum",
     "function": "Cerebellar vermis lobules IV-V; trunk motor control",
     "ad_relevance": "low",
     "notes": "CereN; truncal ataxia"},
    {"index": 111, "name": "Vermis_6",            "hemisphere": "M", "lobe": "Cerebellum",
     "function": "Cerebellar vermis lobule VI; oculomotor and cognitive integration",
     "ad_relevance": "low",
     "notes": "CereN; vermis VI connected to visual cortex"},
    {"index": 112, "name": "Vermis_7",            "hemisphere": "M", "lobe": "Cerebellum",
     "function": "Cerebellar vermis lobule VII; cognitive cerebellar vermis",
     "ad_relevance": "low",
     "notes": "CereN"},
    {"index": 113, "name": "Vermis_8",            "hemisphere": "M", "lobe": "Cerebellum",
     "function": "Cerebellar vermis lobule VIII; sensorimotor vermis",
     "ad_relevance": "low",
     "notes": "CereN"},
    {"index": 114, "name": "Vermis_9",            "hemisphere": "M", "lobe": "Cerebellum",
     "function": "Cerebellar vermis lobule IX; vestibular vermis",
     "ad_relevance": "low",
     "notes": "CereN"},
    {"index": 115, "name": "Vermis_10",           "hemisphere": "M", "lobe": "Cerebellum",
     "function": "Cerebellar vermis lobule X; vestibulocerebellar, equilibrium",
     "ad_relevance": "low",
     "notes": "CereN"},
]

# ═══════════════════════════════════════════════════════════════════
# 2. 腦網路知識（9 個 resting-state networks）
# ═══════════════════════════════════════════════════════════════════
NETWORK_KNOWLEDGE = {
    "DMN": {
        "fullname": "Default Mode Network",
        "roi_indices": [34, 35, 66, 67, 64, 65, 22, 23, 24, 25],
        "function": "Self-referential processing, autobiographical memory, mind-wandering, social cognition, future thinking",
        "ad_description": "Most consistently disrupted network in AD; Aβ preferentially accumulates in DMN hubs (PCC, precuneus, mPFC); FC within DMN is primary imaging biomarker",
        "progression": "PCC-hippocampus FC reduction earliest; spreads to angular gyrus and mPFC in MCI; global DMN collapse in AD",
        # 模型 attention difference (disease - control), 正值 = 疾病端 attention 更高
        "attn_diff": {"NC_vs_AD": -0.07, "NC_vs_MCI": 0.04, "MCI_vs_AD": -0.13},
    },
    "SMN": {
        "fullname": "Sensorimotor Network",
        "roi_indices": [0, 1, 56, 57, 68, 69],
        "function": "Primary motor and somatosensory processing, motor planning, body representation",
        "ad_description": "Relatively preserved in early AD; affected in late-stage with motor symptoms; NC has higher SMN attention than disease stages",
        "progression": "Preserved in MCI; subtle changes in moderate AD; motor symptoms in late AD",
        "attn_diff": {"NC_vs_AD": -0.26, "NC_vs_MCI": -0.16, "MCI_vs_AD": -0.13},
    },
    "VN": {
        "fullname": "Visual Network",
        "roi_indices": [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
        "function": "Primary and higher-order visual processing, visuospatial cognition, object/scene recognition",
        "ad_description": "Shows U-shaped attention pattern: MCI has highest VN attention (compensatory hyperconnectivity), which collapses in AD; PCA variant shows early VN disruption",
        "progression": "MCI: compensatory hyperconnectivity (VN attention highest); AD: FC collapse; PCA variant: primary VN disruption",
        "attn_diff": {"NC_vs_AD": -0.11, "NC_vs_MCI": 0.26, "MCI_vs_AD": -0.41},
    },
    "SN": {
        "fullname": "Salience Network",
        "roi_indices": [28, 29, 30, 31, 32, 33],
        "function": "Salience detection, interoception, switching between DMN and FPN, emotional processing, autonomic regulation",
        "ad_description": "Disrupted early in MCI; NC has significantly higher SN attention than MCI (SN=-0.32 in NC vs MCI); insula and ACC show early tau accumulation",
        "progression": "MCI: insula FC disruption earliest SN change; modulates DMN-FPN switching impaired; AD: severe SN degradation",
        "attn_diff": {"NC_vs_AD": -0.18, "NC_vs_MCI": -0.32, "MCI_vs_AD": 0.12},
    },
    "FPN": {
        "fullname": "Frontoparietal Network",
        "roi_indices": [6, 7, 58, 59, 60, 61],
        "function": "Goal-directed cognition, cognitive control, working memory, top-down attention, flexible task performance",
        "ad_description": "DLPFC-SPL circuit disrupted in MCI; executive function decline correlates with FPN FC reduction; FPN moderates DMN activity",
        "progression": "Subtle FC reduction in MCI between DLPFC and parietal; progressive working memory and executive dysfunction",
        "attn_diff": {"NC_vs_AD": -0.13, "NC_vs_MCI": 0.02, "MCI_vs_AD": -0.16},
    },
    "LN": {
        "fullname": "Limbic Network",
        "roi_indices": [36, 37, 38, 39, 40, 41],
        "function": "Episodic memory encoding/retrieval, emotional processing, fear conditioning, spatial navigation",
        "ad_description": "Hippocampus-entorhinal circuit is primary AD pathology site (Braak Stage I-IV); LN attention is LOWEST in MCI (NC>>MCI), then rises in AD as damage becomes discriminative feature",
        "progression": "Preclinical: entorhinal cortex tau; MCI: hippocampal FC most disrupted (lowest LN attention); AD: widespread limbic damage",
        "attn_diff": {"NC_vs_AD": -0.10, "NC_vs_MCI": -0.33, "MCI_vs_AD": 0.24},
    },
    "VAN": {
        "fullname": "Ventral Attention Network",
        "roi_indices": [10, 11, 14, 15],
        "function": "Stimulus-driven attention reorienting, task-irrelevant salient detection, language-attention integration",
        "ad_description": "Disrupted in MCI affecting attention reorienting; shows positive attention difference in AD (AD>NC); VAN disruption linked to attentional lapses",
        "progression": "MCI: attention reorienting impaired; AD: further VAN dysregulation",
        "attn_diff": {"NC_vs_AD": 0.10, "NC_vs_MCI": -0.08, "MCI_vs_AD": 0.19},
    },
    "BGN": {
        "fullname": "Basal Ganglia Network",
        "roi_indices": [70, 71, 72, 73, 74, 75, 76, 77],
        "function": "Procedural learning, reward processing, motor sequence control, executive function via cortico-striatal loops",
        "ad_description": "Cortico-striatal FC changes in MCI; thalamo-cortical network involved in consciousness and gating; dopaminergic system relatively preserved in typical AD vs Lewy body",
        "progression": "Caudate-DLPFC FC reduction in MCI; thalamic connectivity changes; pallidal involvement in late AD",
        "attn_diff": {"NC_vs_AD": 0.03, "NC_vs_MCI": -0.13, "MCI_vs_AD": 0.15},
    },
    "CereN": {
        "fullname": "Cerebellar Network",
        "roi_indices": list(range(90, 116)),
        "function": "Motor coordination, cognitive processing (Crus I/II), balance, vestibular integration, timing",
        "ad_description": "Cognitive cerebellum (Crus I/II) FC changes with DMN and FPN in AD; motor cerebellum shows late-stage changes; gait and balance impairment in advanced AD",
        "progression": "Crus I/II show FC changes with prefrontal early; motor cerebellum affected in late-stage falls risk",
        "attn_diff": {"NC_vs_AD": 0.11, "NC_vs_MCI": 0.15, "MCI_vs_AD": -0.03},
    },
}

# ═══════════════════════════════════════════════════════════════════
# 3. 疾病階段知識
# ═══════════════════════════════════════════════════════════════════
DISEASE_STAGE_KNOWLEDGE = {
    "NC": {
        "fullname": "Normal Control",
        "criteria": "MMSE >= 27, CDR = 0, no subjective cognitive complaints, no neurological diagnosis",
        "biomarkers": ["Normal hippocampal volume", "Aβ42 CSF > 192 pg/mL", "p-tau < 23 pg/mL", "FDG-PET normal"],
        "fc_characteristics": "Intact DMN FC; strong PCC-hippocampus coupling; balanced SN-DMN switching; preserved FPN-DMN anticorrelation",
    },
    "MCI": {
        "fullname": "Mild Cognitive Impairment",
        "criteria": "MMSE 24-27, CDR = 0.5, subjective and objective memory complaints, preserved daily function, NIA-AA amnestic MCI",
        "biomarkers": ["Hippocampal atrophy (>3%/year)", "Aβ42 CSF borderline 150-192 pg/mL", "p-tau 23-35 pg/mL", "Subtle FDG-PET hypometabolism in PCC"],
        "fc_characteristics": "Reduced PCC-hippocampus FC; VN compensatory hyperconnectivity; SN disruption; LN (limbic) FC most disrupted; conversion rate ~15%/year to AD",
        "subtypes": "sMCI (stable, ADNI-3 predominant) vs pMCI (progressive); sMCI shows NC-like structural features",
    },
    "AD": {
        "fullname": "Alzheimer's Disease",
        "criteria": "MMSE < 24, CDR >= 1, NIA-AA probable AD, significant ADL impairment",
        "biomarkers": ["Severe hippocampal atrophy", "Aβ42 CSF < 150 pg/mL", "p-tau > 35 pg/mL", "FDG-PET hypometabolism in PCC/temporal/parietal", "Positive amyloid PET"],
        "fc_characteristics": "Severe DMN fragmentation; VN FC collapse; LN complete disruption; olfactory and temporal FC severely reduced; global network efficiency decreased",
        "braak_stages": "NFT: Stage I-II (entorhinal/olfactory) → Stage III-IV (hippocampus/amygdala) → Stage V-VI (neocortex)",
    },
}

# ═══════════════════════════════════════════════════════════════════
# 4. 輸出 JSON（無需 Neo4j 也可使用）
# ═══════════════════════════════════════════════════════════════════
def export_to_json(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "roi_knowledge.json"), "w", encoding="utf-8") as f:
        json.dump(ROI_KNOWLEDGE, f, ensure_ascii=False, indent=2)
    print(f"  ✅ roi_knowledge.json  ({len(ROI_KNOWLEDGE)} ROIs)")

    with open(os.path.join(output_dir, "network_knowledge.json"), "w", encoding="utf-8") as f:
        json.dump(NETWORK_KNOWLEDGE, f, ensure_ascii=False, indent=2)
    print(f"  ✅ network_knowledge.json  ({len(NETWORK_KNOWLEDGE)} networks)")

    with open(os.path.join(output_dir, "disease_stage_knowledge.json"), "w", encoding="utf-8") as f:
        json.dump(DISEASE_STAGE_KNOWLEDGE, f, ensure_ascii=False, indent=2)
    print(f"  ✅ disease_stage_knowledge.json")


# ═══════════════════════════════════════════════════════════════════
# 5. 寫入 Neo4j
# ═══════════════════════════════════════════════════════════════════
def build_neo4j_graph():
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("  ⚠️  neo4j 套件未安裝，請執行: pip install neo4j")
        return

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        # ── 清空舊資料（開發時使用）──────────────────────────────
        session.run("MATCH (n) DETACH DELETE n")
        print("  🗑️  清空舊 graph")

        # ── 建立 Constraint / Index ───────────────────────────────
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:ROI) REQUIRE r.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:BrainNetwork) REQUIRE n.abbr IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:DiseaseStage) REQUIRE d.name IS UNIQUE")

        # ── 寫入 DiseaseStage 節點 ────────────────────────────────
        for stage, info in DISEASE_STAGE_KNOWLEDGE.items():
            session.run("""
                MERGE (d:DiseaseStage {name: $name})
                SET d.fullname          = $fullname,
                    d.criteria          = $criteria,
                    d.fc_characteristics = $fc_char
            """, name=stage, fullname=info["fullname"],
                 criteria=info["criteria"], fc_char=info["fc_characteristics"])
        print(f"  ✅ DiseaseStage 節點  (3)")

        # ── 寫入 BrainNetwork 節點 ────────────────────────────────
        for abbr, info in NETWORK_KNOWLEDGE.items():
            session.run("""
                MERGE (n:BrainNetwork {abbr: $abbr})
                SET n.fullname        = $fullname,
                    n.function        = $function,
                    n.ad_description  = $ad_desc,
                    n.progression     = $progression
            """, abbr=abbr, fullname=info["fullname"],
                 function=info["function"], ad_desc=info["ad_description"],
                 progression=info["progression"])

            # BrainNetwork → ATTENTION_DIFF → DiseaseStage（存三個 task 的差值）
            for task_str, diff_val in info["attn_diff"].items():
                parts  = task_str.split("_vs_")
                ctrl   = parts[0]
                dis    = parts[1]
                # 邊方向：network → disease_stage，以 task 標記
                session.run("""
                    MATCH (n:BrainNetwork {abbr: $abbr})
                    MATCH (d:DiseaseStage {name: $disease})
                    MERGE (n)-[r:ATTENTION_DIFF {task: $task}]->(d)
                    SET r.diff_value    = $diff,
                        r.control_stage = $ctrl,
                        r.interpretation = CASE
                            WHEN $diff > 0.15 THEN 'disease_dominant'
                            WHEN $diff < -0.15 THEN 'control_dominant'
                            ELSE 'balanced'
                        END
                """, abbr=abbr, disease=dis, task=task_str,
                     diff=diff_val, ctrl=ctrl)
        print(f"  ✅ BrainNetwork 節點 + ATTENTION_DIFF 邊  ({len(NETWORK_KNOWLEDGE)} networks)")

        # ── 寫入 ROI 節點 + BELONGS_TO 邊 ────────────────────────
        roi_to_net = {
            roi_i: abbr
            for abbr, info in NETWORK_KNOWLEDGE.items()
            for roi_i in info["roi_indices"]
        }
        for roi in ROI_KNOWLEDGE:
            session.run("""
                MERGE (r:ROI {name: $name})
                SET r.index        = $index,
                    r.hemisphere   = $hemisphere,
                    r.lobe         = $lobe,
                    r.function     = $function,
                    r.ad_relevance = $ad_relevance,
                    r.notes        = $notes
            """, **{k: roi[k] for k in
                   ["name", "index", "hemisphere", "lobe",
                    "function", "ad_relevance", "notes"]})

            if roi["index"] in roi_to_net:
                net_abbr = roi_to_net[roi["index"]]
                session.run("""
                    MATCH (r:ROI  {name: $roi_name})
                    MATCH (n:BrainNetwork {abbr: $abbr})
                    MERGE (r)-[:BELONGS_TO]->(n)
                """, roi_name=roi["name"], abbr=net_abbr)
        print(f"  ✅ ROI 節點 + BELONGS_TO 邊  ({len(ROI_KNOWLEDGE)} ROIs)")

    driver.close()
    print("\n  Graph build complete.")
    print("  驗證：在 Neo4j Browser 執行")
    print("    MATCH (r:ROI)-[:BELONGS_TO]->(n:BrainNetwork) RETURN r,n LIMIT 50")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    JSON_OUT = "/home/wei-chi/Data/script/results/knowledge_base"
    print("📚 匯出知識庫 JSON...")
    export_to_json(JSON_OUT)

    print("\n🔗 寫入 Neo4j...")
    build_neo4j_graph()

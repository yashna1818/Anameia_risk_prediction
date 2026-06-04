import os
import io
import time
import gzip
import warnings
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, mean_absolute_error, r2_score, mean_squared_error
)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# -----------------------------------------------------------------------------
# GLOBAL CONFIG & WARNINGS
# -----------------------------------------------------------------------------
warnings.filterwarnings('ignore')
np.random.seed(42)

app = FastAPI(
    title="Anemia Risk Prediction & Precision Nutrition Pipeline",
    description="Leakage-free XGBoost prediction with VCF-based PRS and metabolic pathway analysis."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to datasets in the system
GWAS_PATH = "/Users/yashna/IDPMODEL/FINAL_CLEAN_GWAS_400.csv"
NHANES_PATH = "/Users/yashna/IDPMODEL/phenotype_diet NHANES.csv"

# Global Model Store
MODELS = {
    "classifier": None,
    "regressor": None,
    "scaler": None,
    "feature_names": [],
    "snp_ids": [],
    "gwas_sig": None,
    "snp_bio": None,
    "nhanes_median": {},
    "prs_p25": 0.0,
    "prs_p75": 0.0,
    "prs_le": None,
    "is_trained": False
}

# -----------------------------------------------------------------------------
# NUTRIGENOMICS KNOWLEDGE BASE
# -----------------------------------------------------------------------------
GENE_NUTRITION_KB = {
    "TMPRSS6": {
        "pathway": "Hepcidin Regulation (Matriptase-2)",
        "mechanism": "TMPRSS6 cleaves hemojuvelin to suppress hepcidin. Risk allele -> chronically elevated hepcidin -> ferroportin degradation -> iron trapped in gut enterocytes and macrophages -> iron-refractory iron deficiency anemia (IRIDA).",
        "pathway_class": "hepcidin",
        "increase": [
            {"nutrient": "Heme iron (meat/fish)", "evidence": "Strong", "rationale": "Heme iron bypasses hepcidin-blocked ferroportin via a separate transporter"},
            {"nutrient": "Vitamin C (≥200mg with meals)", "evidence": "Strong", "rationale": "Reduces Fe³⁺ to Fe²⁺; maximises what little non-heme iron passes the hepcidin block"},
            {"nutrient": "Anti-inflammatory foods (omega-3, turmeric)", "evidence": "Moderate", "rationale": "Reduce IL-6-driven hepcidin elevation"},
        ],
        "avoid": [
            {"nutrient": "Tea/coffee within 1h of meals", "evidence": "Strong", "rationale": "Tannins reduce non-heme absorption by up to 60% — critical when hepcidin already limits it"},
            {"nutrient": "Calcium supplements with meals", "evidence": "Moderate", "rationale": "Competes with iron at DMT1 transporter"},
            {"nutrient": "Unsoaked legumes", "evidence": "Moderate", "rationale": "Phytates further reduce already-impaired absorption"},
        ],
        "clinical_flag": True,
        "clinical_note": "IRIDA — oral iron may be ineffective. IV iron infusion may be required. Measure serum hepcidin.",
    },
    "HAMP": {
        "pathway": "Hepcidin Production (Master Iron Hormone)",
        "mechanism": "HAMP encodes hepcidin itself. Gain-of-function -> excess hepcidin -> ferroportin degradation -> iron sequestration. Anti-inflammatory interventions can reduce hepcidin production.",
        "pathway_class": "hepcidin",
        "increase": [
            {"nutrient": "Omega-3 fatty acids", "evidence": "Strong", "rationale": "DHA/EPA reduce IL-6, the primary inducer of hepcidin transcription"},
            {"nutrient": "Curcumin (turmeric)", "evidence": "Moderate", "rationale": "Reduces NF-κB-driven hepcidin expression in hepatocytes"},
            {"nutrient": "Vitamin C with every iron meal", "evidence": "Strong", "rationale": "Maximises absorption against hepcidin block"},
        ],
        "avoid": [
            {"nutrient": "Tea/coffee within 1h of meals", "evidence": "Strong", "rationale": "Worsens already-impaired absorption"},
            {"nutrient": "Ultra-processed foods", "evidence": "Moderate", "rationale": "Promote systemic inflammation -> raise IL-6 -> raise hepcidin"},
        ],
        "clinical_flag": True,
        "clinical_note": "Measure serum hepcidin and ferritin. Anti-inflammatory diet is adjunct to clinical management.",
    },
    "CYBRD1": {
        "pathway": "Duodenal Iron Reduction (DcytB, Fe³⁺ → Fe²⁺)",
        "mechanism": "CYBRD1/DcytB converts insoluble dietary Fe³⁺ to absorbable Fe²⁺ at the brush border. Variant -> less efficient reduction -> less iron crosses into enterocytes. Vitamin C performs the identical chemical reaction.",
        "pathway_class": "absorption",
        "increase": [
            {"nutrient": "Vitamin C with EVERY iron meal", "evidence": "Strong", "rationale": "Chemically replaces DcytB function — converts Fe³⁺ to Fe²⁺ in the gut lumen"},
            {"nutrient": "Amla / lemon / orange", "evidence": "Strong", "rationale": "Highest-Vit-C Indian foods; must accompany iron sources at same meal"},
            {"nutrient": "Fermented foods", "evidence": "Moderate", "rationale": "Reduce phytate content and lower gut pH, favouring Fe²⁺ stability"},
        ],
        "avoid": [
            {"nutrient": "Tea/coffee within 1h of meals", "evidence": "Strong", "rationale": "Tannins oxidise Fe²⁺ back to Fe³⁺, directly reversing DcytB/Vit-C reduction"},
            {"nutrient": "Antacids", "evidence": "Strong", "rationale": "Raise gut pH, shifting equilibrium back toward Fe³⁺ (insoluble form)"},
        ],
        "clinical_flag": False,
        "clinical_note": "Vitamin C supplementation (250mg with meals) may be more effective than extra dietary iron alone.",
    },
    "SLC11A2": {
        "pathway": "Intestinal Iron Uptake (DMT1 Transporter)",
        "mechanism": "SLC11A2/DMT1 transports Fe²⁺ from the gut lumen into enterocytes. Variants reduce transport efficiency for non-heme iron. Heme iron uses a separate, DMT1-independent pathway.",
        "pathway_class": "absorption",
        "increase": [
            {"nutrient": "Heme iron (red meat, fish)", "evidence": "Strong", "rationale": "Heme iron enters enterocytes via HCP1/PCFT, completely bypassing defective DMT1"},
            {"nutrient": "Vitamin C with plant iron", "evidence": "Strong", "rationale": "Maximises Fe²⁺ availability to compete for reduced DMT1 capacity"},
            {"nutrient": "Fermented foods (idli, dosa, kimchi)", "evidence": "Moderate", "rationale": "Fermentation reduces phytates and produces organic acids that chelate iron as Fe²⁺"},
        ],
        "avoid": [
            {"nutrient": "Phytate-heavy unsoaked legumes", "evidence": "Strong", "rationale": "Phytates chelate Fe²⁺ before it can enter DMT1 — doubly harmful with reduced transport"},
            {"nutrient": "Zinc supplements at same time", "evidence": "Moderate", "rationale": "Zinc and iron share DMT1; high zinc directly competes for already-limited transport"},
        ],
        "clinical_flag": False,
        "clinical_note": "If vegetarian, consider IV iron in consultation with a haematologist if oral supplementation is inadequate.",
    },
    "HFE": {
        "pathway": "Iron Sensing and Homeostasis (HFE/TFR1 complex)",
        "mechanism": "HFE interacts with TFR1 in duodenal crypts to sense systemic iron and regulate hepcidin. C282Y/H63D variants impair sensing, causing either iron overload (homozygous C282Y) or milder dysregulation (H63D). Direction of effect MUST be confirmed with serum ferritin + transferrin saturation.",
        "pathway_class": "iron_sensing",
        "increase": [
            {"nutrient": "Polyphenols (green tea, berries, dark chocolate)", "evidence": "Strong", "rationale": "Inhibit excess iron absorption; antioxidant protection against iron-induced oxidative stress"},
            {"nutrient": "Cruciferous vegetables", "evidence": "Moderate", "rationale": "Support liver detoxification pathways relevant to iron metabolism"},
        ],
        "avoid": [
            {"nutrient": "Iron supplements without testing", "evidence": "Strong", "rationale": "Risk of dangerous iron overload in C282Y homozygotes — MUST test ferritin first"},
            {"nutrient": "Excess vitamin C supplements", "evidence": "Moderate", "rationale": "Can accelerate iron absorption in overload-prone individuals"},
            {"nutrient": "Alcohol", "evidence": "Strong", "rationale": "Directly hepatotoxic in iron overload; also increases iron absorption"},
        ],
        "clinical_flag": True,
        "clinical_note": "CRITICAL: Test serum ferritin + transferrin saturation BEFORE recommending any iron. Opposite interventions needed for overload vs deficiency.",
    },
    "TF": {
        "pathway": "Blood Iron Transport (Transferrin)",
        "mechanism": "TF carries iron in plasma from absorption sites and macrophages to bone marrow. Variants reduce TF expression or iron-binding affinity -> less iron delivered per unit time to erythroblasts.",
        "pathway_class": "transferrin",
        "increase": [
            {"nutrient": "High dietary iron (maximise transferrin saturation)", "evidence": "Strong", "rationale": "More iron loaded onto each TF molecule compensates for reduced delivery efficiency"},
            {"nutrient": "Vitamin C with iron meals", "evidence": "Strong", "rationale": "Maximises iron absorption to keep transferrin saturation high"},
            {"nutrient": "Copper (nuts, seeds, legumes)", "evidence": "Moderate", "rationale": "Ceruloplasmin (copper enzyme) is required to oxidise Fe²⁺ to Fe³⁺ for TF loading"},
            {"nutrient": "Vitamin A (eggs, dairy, carrots)", "evidence": "Moderate", "rationale": "Mobilises iron from storage to transferrin; deficiency mimics iron deficiency"},
        ],
        "avoid": [
            {"nutrient": "Tea/coffee within 1h", "evidence": "Strong", "rationale": "Reduces iron to load onto transferrin"},
        ],
        "clinical_flag": False,
        "clinical_note": "Transferrin saturation <16% + low ferritin confirms functional iron deficiency. Monitor both.",
    },
    "SLC40A1": {
        "pathway": "Cellular Iron Export (Ferroportin)",
        "mechanism": "SLC40A1/ferroportin is the sole known iron exporter from cells. Loss-of-function -> iron absorbed by enterocytes cannot be exported to blood. Gain-of-function (hepcidin-resistant) -> ferroportin disease.",
        "pathway_class": "transferrin",
        "increase": [
            {"nutrient": "High dietary iron + Vitamin C", "evidence": "Strong", "rationale": "Saturate the reduced export capacity with maximum available iron"},
        ],
        "avoid": [
            {"nutrient": "Iron supplements without monitoring", "evidence": "Strong", "rationale": "Gain-of-function variants cause iron overload — MUST confirm variant type clinically"},
        ],
        "clinical_flag": True,
        "clinical_note": "Ferroportin disease diagnosis requires genetic testing. Direction of intervention (increase vs restrict iron) depends on variant class.",
    },
    "TFR2": {
        "pathway": "Iron Sensing via Transferrin Receptor 2",
        "mechanism": "TFR2 in hepatocytes senses diferric transferrin and signals for hepcidin production. Loss-of-function -> impaired iron sensing -> inappropriately low hepcidin -> can cause HFE2-type hereditary hemochromatosis OR altered erythroid iron utilisation.",
        "pathway_class": "transferrin",
        "increase": [
            {"nutrient": "Copper (liver, nuts, seeds)", "evidence": "Moderate", "rationale": "Supports ceruloplasmin-mediated iron loading onto transferrin, feeding TFR2 sensing pathway"},
            {"nutrient": "Vitamin A", "evidence": "Moderate", "rationale": "Supports iron mobilisation and transferrin saturation for accurate sensing"},
        ],
        "avoid": [
            {"nutrient": "Iron supplements without testing", "evidence": "Strong", "rationale": "TFR2 loss-of-function can cause hemochromatosis — test ferritin first"},
        ],
        "clinical_flag": True,
        "clinical_note": "Test serum ferritin + transferrin saturation. TFR2 mutations can cause hereditary hemochromatosis type 3.",
    },
    "EPO": {
        "pathway": "Erythropoietin — Bone Marrow Stimulation",
        "mechanism": "EPO from the kidney drives erythroid progenitor proliferation and differentiation. Low EPO -> fewer RBCs regardless of iron status. Diet supports EPO effectiveness but cannot compensate for severely reduced EPO production.",
        "pathway_class": "erythropoiesis",
        "increase": [
            {"nutrient": "Iron (adequate stores)", "evidence": "Strong", "rationale": "EPO drives erythroblast proliferation only if iron is available for Hb synthesis"},
            {"nutrient": "Folate", "evidence": "Strong", "rationale": "Required for rapid DNA synthesis in EPO-stimulated erythroblast expansion"},
            {"nutrient": "Vitamin B12", "evidence": "Strong", "rationale": "Works with folate for erythroblast DNA synthesis; deficiency blocks EPO response"},
        ],
        "avoid": [
            {"nutrient": "Alcohol", "evidence": "Strong", "rationale": "Directly suppresses EPO production in the kidney"},
            {"nutrient": "Smoking", "evidence": "Moderate", "rationale": "Chronic hypoxia from smoking dysregulates EPO; oxidative stress damages erythroblasts"},
        ],
        "clinical_flag": False,
        "clinical_note": "Measure serum EPO. If low, investigate chronic kidney disease or inflammation as primary cause.",
    },
    "GATA1": {
        "pathway": "Erythroid Transcription (GATA1 Master Regulator)",
        "mechanism": "GATA1 is the master transcription factor for erythroid differentiation. Variants impair RBC maturation at the precursor stage, reducing functional RBC output even when iron and EPO are adequate.",
        "pathway_class": "erythropoiesis",
        "increase": [
            {"nutrient": "Folate", "evidence": "Strong", "rationale": "Required for erythroblast DNA replication during GATA1-driven differentiation"},
            {"nutrient": "Vitamin B12", "evidence": "Strong", "rationale": "Cofactor for methionine synthase; deficiency causes maturation arrest mimicking GATA1 impairment"},
            {"nutrient": "Iron", "evidence": "Strong", "rationale": "Terminal erythroid differentiation requires iron for haemoglobin filling of maturing erythroblasts"},
            {"nutrient": "Riboflavin (B2)", "evidence": "Moderate", "rationale": "Flavoproteins support erythroid mitochondrial function during differentiation"},
        ],
        "avoid": [],
        "clinical_flag": False,
        "clinical_note": "GATA1 variants can cause X-linked thrombocytopenia or dyserythropoietic anemia — haematology review recommended.",
    },
    "ALAS2": {
        "pathway": "Haem Biosynthesis Step 1 (Pyridoxine-Dependent)",
        "mechanism": "ALAS2 condenses succinyl-CoA + glycine -> δ-aminolevulinic acid, the first and rate-limiting step of haem synthesis. Requires Vitamin B6 (pyridoxal phosphate) as cofactor. Variants cause X-linked sideroblastic anemia — iron present but cannot be incorporated into haem.",
        "pathway_class": "haem_synthesis",
        "increase": [
            {"nutrient": "Vitamin B6 (≥25mg/day clinically)", "evidence": "Strong", "rationale": "B6 is the essential ALAS2 cofactor — supplementation is the specific clinical treatment for ALAS2-related sideroblastic anemia"},
            {"nutrient": "B6-rich foods daily", "evidence": "Strong", "rationale": "Chicken breast, fish, banana, potato (with skin), pistachios — all meals should include one"},
            {"nutrient": "Iron (maintains substrate supply)", "evidence": "Moderate", "rationale": "Adequate iron ensures substrate is available when ALAS2 produces porphyrin"},
        ],
        "avoid": [
            {"nutrient": "Alcohol", "evidence": "Strong", "rationale": "Inhibits ALAS2 directly and antagonises B6 metabolism — major sideroblastic anemia trigger"},
            {"nutrient": "Isoniazid / pyrazinamide", "evidence": "Strong", "rationale": "These TB drugs are B6 antagonists — clinically dangerous for ALAS2 variant carriers"},
        ],
        "clinical_flag": True,
        "clinical_note": "X-linked sideroblastic anemia. B6 supplementation (50–200mg/day) often prescribed. Bone marrow biopsy diagnostic. Genetic counselling recommended.",
    },
    "FECH": {
        "pathway": "Haem Biosynthesis Step 8 — Ferrochelatase (Final Step)",
        "mechanism": "FECH inserts Fe²⁺ into protoporphyrin IX to form haem — the final step. Loss-of-function -> protoporphyrin accumulates, iron cannot be incorporated into Hb. Causes erythropoietic protoporphyria (EPP).",
        "pathway_class": "haem_synthesis",
        "increase": [
            {"nutrient": "Iron", "evidence": "Moderate", "rationale": "Maintains Fe²⁺ supply for residual FECH activity"},
            {"nutrient": "Vitamin B6", "evidence": "Moderate", "rationale": "Supports upstream haem synthesis steps that feed into FECH"},
        ],
        "avoid": [
            {"nutrient": "Sun exposure (clinical context)", "evidence": "Strong", "rationale": "In EPP, protoporphyrin accumulation causes phototoxicity — not dietary but lifestyle-critical"},
        ],
        "clinical_flag": True,
        "clinical_note": "Erythropoietic protoporphyria — haematology + dermatology referral. RBC protoporphyrin measurement needed.",
    },
    "MTHFR": {
        "pathway": "Folate Metabolism (MTHFR Enzyme — Methylation Cycle)",
        "mechanism": "MTHFR converts 5,10-methyleneTHF -> 5-methylTHF (active folate). C677T reduces activity 70% (homozygous). Very common in South Asians. -> Impaired DNA synthesis in erythroblasts -> megaloblastic tendency. -> Elevated homocysteine -> vascular risk.",
        "pathway_class": "folate_b12",
        "increase": [
            {"nutrient": "Methylfolate (5-MTHF supplement)", "evidence": "Strong", "rationale": "Bypasses the defective MTHFR step — bioactive form does not require MTHFR conversion"},
            {"nutrient": "Dark leafy greens daily", "evidence": "Strong", "rationale": "Natural folate (spinach, methi, palak) provides 5-MTHF directly with minimal conversion needed"},
            {"nutrient": "Vitamin B12", "evidence": "Strong", "rationale": "Required by methionine synthase downstream of MTHFR; B12 deficiency traps folate as 5-MTHF"},
            {"nutrient": "Riboflavin (B2)", "evidence": "Moderate", "rationale": "FAD cofactor for MTHFR — riboflavin supplementation partially restores C677T MTHFR activity"},
        ],
        "avoid": [
            {"nutrient": "Standard folic acid alone", "evidence": "Strong", "rationale": "Requires MTHFR to convert to active form — insufficient alone for C677T carriers"},
            {"nutrient": "Alcohol", "evidence": "Strong", "rationale": "Directly depletes folate stores and inhibits folate absorption"},
        ],
        "clinical_flag": False,
        "clinical_note": "Recommend methylfolate supplement (400–800mcg/day). Check plasma homocysteine level. Test MTHFR genotype (C677T/A1298C) for clinical classification.",
    },
    "MTR": {
        "pathway": "Methionine Synthase (B12 + Folate Dependent)",
        "mechanism": "MTR (methionine synthase) catalyses homocysteine -> methionine using 5-MTHF and B12. Deficiency -> methylfolate trap -> megaloblastic anemia. Requires both B12 AND active folate.",
        "pathway_class": "folate_b12",
        "increase": [
            {"nutrient": "Vitamin B12", "evidence": "Strong", "rationale": "Direct cofactor for MTR — essential for enzyme function"},
            {"nutrient": "Methylfolate (5-MTHF)", "evidence": "Strong", "rationale": "Active folate substrate for MTR; standard folic acid may be insufficient"},
            {"nutrient": "Riboflavin (B2)", "evidence": "Moderate", "rationale": "Supports MTHFR activity upstream of MTR"},
        ],
        "avoid": [
            {"nutrient": "Nitrous oxide (N₂O)", "evidence": "Strong", "rationale": "Irreversibly oxidises B12 cobalt centre, inactivating MTR — dangerous for MTR variant carriers"},
            {"nutrient": "Alcohol", "evidence": "Strong", "rationale": "Depletes B12 and folate simultaneously"},
        ],
        "clinical_flag": True,
        "clinical_note": "Measure serum B12, folate, and plasma homocysteine. B12 deficiency + MTR variant causes severe megaloblastic anemia.",
    },
    "G6PD": {
        "pathway": "Pentose Phosphate Pathway (RBC Oxidative Protection)",
        "mechanism": "G6PD generates NADPH to recycle glutathione, the main RBC antioxidant. Deficiency -> RBC membrane oxidative damage -> haemolytic episodes when exposed to oxidative triggers. Most common enzyme disorder worldwide; especially prevalent in malaria-endemic South Asia and Africa.",
        "pathway_class": "rbc_membrane",
        "increase": [
            {"nutrient": "Antioxidant-rich foods", "evidence": "Strong", "rationale": "Blueberries, jamun, turmeric, amla protect RBCs from oxidative damage when G6PD is deficient"},
            {"nutrient": "Folate", "evidence": "Strong", "rationale": "Critical for RBC regeneration after haemolytic episodes"},
            {"nutrient": "Riboflavin (B2)", "evidence": "Moderate", "rationale": "FAD required for glutathione reductase — partially compensates G6PD-deficient NADPH deficit"},
        ],
        "avoid": [
            {"nutrient": "Fava beans (ABSOLUTE AVOID)", "evidence": "Strong", "rationale": "Vicine/convicine in fava beans directly trigger acute haemolytic crisis in G6PD-deficient patients"},
            {"nutrient": "High-dose Vitamin C supplements", "evidence": "Strong", "rationale": "Pro-oxidant at pharmacological doses in G6PD-deficient RBCs"},
            {"nutrient": "Red wine / sulphites", "evidence": "Moderate", "rationale": "Can trigger haemolysis in sensitive G6PD-deficient individuals"},
        ],
        "clinical_flag": True,
        "clinical_note": "CRITICAL MEDICATION ALERT: Avoid primaquine, dapsone, rasburicase, nitrofurantoin — all trigger haemolysis. Carry G6PD deficiency alert card.",
    },
    "PIEZO1": {
        "pathway": "RBC Volume Regulation (Mechanosensory Ca²⁺ Channel)",
        "mechanism": "PIEZO1 gain-of-function -> excessive Ca²⁺ entry -> Gardos channel activation (K⁺ and water efflux) -> RBC dehydration -> stiff, dense RBCs -> splenic destruction. Common in West and East African populations; also found in South Asian individuals.",
        "pathway_class": "rbc_membrane",
        "increase": [
            {"nutrient": "Hydration (2.5–3L water/day)", "evidence": "Strong", "rationale": "Reduces blood viscosity and supports less-deformable RBC transit through spleen"},
            {"nutrient": "Omega-3 fatty acids", "evidence": "Moderate", "rationale": "DHA incorporates into RBC membranes, improving flexibility and deformability"},
            {"nutrient": "Folate", "evidence": "Strong", "rationale": "Supports RBC regeneration to replace splenicly cleared dehydrated cells"},
            {"nutrient": "Magnesium-rich foods", "evidence": "Moderate", "rationale": "Mg²⁺ modulates ion channel activity and may partially offset Ca²⁺ accumulation"},
        ],
        "avoid": [
            {"nutrient": "Dehydrating drinks (excess caffeine/alcohol)", "evidence": "Moderate", "rationale": "Worsen RBC dehydration in already Gardos-channel-activated cells"},
        ],
        "clinical_flag": False,
        "clinical_note": "PIEZO1 gain-of-function (E756del in African populations) associated with hereditary xerocytosis. Splenomegaly possible.",
    },
    "ATP2B4": {
        "pathway": "RBC Calcium Homeostasis (PMCA4 Calcium Pump)",
        "mechanism": "ATP2B4/PMCA4 extrudes Ca²⁺ from RBCs. Loss-of-function -> intracellular Ca²⁺ accumulates -> Gardos channel activation -> K⁺/water efflux -> RBC dehydration -> increased splenic haemolysis. Mechanistically identical to PIEZO1 gain-of-function downstream of Ca²⁺ elevation.",
        "pathway_class": "rbc_membrane",
        "increase": [
            {"nutrient": "Hydration (2.5–3L/day)", "evidence": "Strong", "rationale": "Reduces haemoconcentration and spleen-mediated RBC clearance"},
            {"nutrient": "Omega-3 (fatty fish, flaxseed)", "evidence": "Moderate", "rationale": "DHA in RBC membranes improves deformability; EPA reduces inflammatory RBC destruction"},
            {"nutrient": "Magnesium (nuts, seeds, greens)", "evidence": "Moderate", "rationale": "Competes with Ca²⁺ for intracellular binding sites; may partially buffer Ca²⁺ accumulation"},
        ],
        "avoid": [
            {"nutrient": "Caffeine excess", "evidence": "Weak", "rationale": "May modulate intracellular Ca²⁺ signalling in sensitive individuals"},
        ],
        "clinical_flag": False,
        "clinical_note": "Measure RBC osmotic fragility if clinical haemolysis suspected.",
    },
    "ACSL3": {
        "pathway": "RBC Membrane Lipid Metabolism (Long-chain Fatty Acid Activation)",
        "mechanism": "ACSL3 activates long-chain fatty acids (C16–C22) by conjugating to CoA for membrane phospholipid incorporation. RBC membranes are ~50% lipid; phospholipid composition determines deformability and RBC lifespan. Variants -> suboptimal membrane lipid composition -> stiffer RBCs -> increased splenic destruction -> lower Hb. (Strongest GWAS signal in dataset.)",
        "pathway_class": "rbc_membrane",
        "increase": [
            {"nutrient": "Pre-formed EPA/DHA (fatty fish)", "evidence": "Strong", "rationale": "Direct membrane-ready omega-3 that ACSL3 can incorporate without further conversion"},
            {"nutrient": "Fatty fish 2×/week (salmon, mackerel, sardines, hilsa)", "evidence": "Strong", "rationale": "Best source of EPA/DHA — provides C20:5 and C22:6 directly for membrane phospholipids"},
            {"nutrient": "Ground flaxseed (1 tbsp/day)", "evidence": "Moderate", "rationale": "ALA source; partially converted to EPA — less efficient than fish but beneficial"},
        ],
        "avoid": [
            {"nutrient": "Trans-fats (biscuits, namkeen, fried snacks)", "evidence": "Strong", "rationale": "Trans-fatty acids compete with omega-3 for membrane incorporation, worsening RBC deformability"},
            {"nutrient": "Refined seed oils high in omega-6", "evidence": "Moderate", "rationale": "Excess omega-6 shifts membrane composition away from omega-3, reduces RBC flexibility"},
        ],
        "clinical_flag": False,
        "clinical_note": "RBC membrane fatty acid profiling available in specialised labs if severe unexplained haemolytic anemia.",
    },
    "ACSL3P1": {
        "pathway": "RBC Membrane Lipid Metabolism (ACSL3 Regulatory Locus — Strongest GWAS Signal)",
        "mechanism": "ACSL3P1 is a pseudogene locus adjacent to ACSL3 that acts as a regulatory element (enhancer/silencer) modulating ACSL3 expression in erythroid cells. Effect on RBC membrane lipid composition is identical to ACSL3 above. Beta>1.1 in your GWAS — the most functionally important variant for this patient.",
        "pathway_class": "rbc_membrane",
        "increase": [
            {"nutrient": "Fatty fish 2×/week (pre-formed EPA/DHA)", "evidence": "Strong", "rationale": "ACSL3-regulatory variant -> prioritise pre-formed omega-3 vs plant conversion"},
            {"nutrient": "Ground flaxseed + walnuts daily", "evidence": "Moderate", "rationale": "Combined ALA sources provide partial EPA/DHA precursor even with reduced ACSL3"},
            {"nutrient": "Algal oil (DHA from algae)", "evidence": "Strong", "rationale": "For vegetarians: direct DHA bypasses ACSL3 conversion limitation"},
        ],
        "avoid": [
            {"nutrient": "Trans-fats completely", "evidence": "Strong", "rationale": "Highest priority avoidance — directly worsens membrane composition in ACSL3-compromised RBCs"},
            {"nutrient": "Processed and ultra-processed foods", "evidence": "Strong", "rationale": "Contain both trans-fats and pro-inflammatory omega-6 oils"},
        ],
        "clinical_flag": False,
        "clinical_note": "This is the highest-impact variant in your GWAS. RBC membrane omega-3 status is the primary nutritional target.",
    },
    "FADS1": {
        "pathway": "Omega-3 Conversion (Delta-5 Desaturase)",
        "mechanism": "FADS1 converts plant ALA (18:3n-3) -> EPA (20:5n-3), the precursor to DHA. rs174537 risk allele reduces conversion by 40–80%. Flaxseed/walnuts provide ALA that cannot be efficiently converted in these patients.",
        "pathway_class": "rbc_membrane",
        "increase": [
            {"nutrient": "Pre-formed EPA/DHA from fatty fish", "evidence": "Strong", "rationale": "BYPASSES the defective FADS1 desaturation step entirely — the correct solution for this variant"},
            {"nutrient": "Algal DHA/EPA supplement", "evidence": "Strong", "rationale": "Vegan-friendly pre-formed omega-3 from microalgae — the original source before the fish food chain"},
        ],
        "avoid": [
            {"nutrient": "Relying on flaxseed/walnuts ALONE", "evidence": "Strong", "rationale": "ALA from plants requires FADS1 for EPA conversion — ineffective alone for this genotype"},
        ],
        "clinical_flag": False,
        "clinical_note": "FADS1 rs174537 testing available in nutrigenomics panels. Confirm genotype for precision dosing of EPA/DHA.",
    },
    "GMPR": {
        "pathway": "RBC Purine Nucleotide Cycle (ATP Maintenance)",
        "mechanism": "GMPR catalyses GMP->IMP in the purine interconversion cycle — essential for maintaining RBC ATP levels. Mature RBCs cannot synthesise purines de novo; GMPR variants -> ATP depletion -> cytoskeletal failure -> shape change -> splenic destruction -> haemolytic anemia.",
        "pathway_class": "rbc_energy",
        "increase": [
            {"nutrient": "Folate (dark leafy greens, lentils)", "evidence": "Strong", "rationale": "Folate provides 10-formyl-THF for purine synthesis, supporting the nucleotide pool that feeds GMPR"},
            {"nutrient": "Vitamin B12", "evidence": "Moderate", "rationale": "Works with folate in one-carbon metabolism feeding purine precursor synthesis"},
            {"nutrient": "Adequate hydration (2.5L/day)", "evidence": "Moderate", "rationale": "Dehydration concentrates RBCs and worsens ATP-depletion-related shape abnormalities"},
            {"nutrient": "Ribose-containing foods (whole grains)", "evidence": "Weak", "rationale": "Pentose phosphate pathway provides ribose-5-phosphate for nucleotide salvage in RBCs"},
        ],
        "avoid": [
            {"nutrient": "Prolonged fasting", "evidence": "Moderate", "rationale": "Fasting depletes RBC glycolytic substrate and nucleotide precursors, worsening GMPR-related ATP deficit"},
            {"nutrient": "Alcohol", "evidence": "Strong", "rationale": "Directly depletes folate and purines, compounding GMPR-related nucleotide depletion"},
        ],
        "clinical_flag": False,
        "clinical_note": "RBC purine nucleotide profile in specialised labs if unexplained haemolysis with folate/B12 supplementation.",
    },
    "RAB11FIP3": {
        "pathway": "Endosomal Recycling — Transferrin Receptor (TFR1) Trafficking",
        "mechanism": "RAB11FIP3 mediates recycling of TFR1 from endosomes back to the cell surface after each iron import cycle. Impaired recycling -> fewer surface TFR1 -> fewer iron import cycles per erythroblast -> insufficient Hb synthesis.",
        "pathway_class": "transferrin",
        "increase": [
            {"nutrient": "High iron foods + Vitamin C at every meal", "evidence": "Strong", "rationale": "Maximise iron delivered per available TFR1 cycle — compensates for reduced receptor recycling efficiency"},
        ],
        "avoid": [
            {"nutrient": "Tea/coffee within 1h", "evidence": "Strong", "rationale": "Reduces iron absorption, wasting already-limited TFR1 cycles on low-iron delivery"},
        ],
        "clinical_flag": False,
        "clinical_note": "Iron studies (serum iron, TIBC, ferritin) help quantify the functional impact on iron delivery.",
    }
}

PATHWAY_KB = {}
for gene_key, kb in GENE_NUTRITION_KB.items():
    PATHWAY_KB[gene_key] = {
        'pathway': kb['pathway'],
        'mechanism': kb['mechanism'],
        'anemia_link': f"Pathway class '{kb['pathway_class']}' affected. " + (kb.get('clinical_note', '') or 'See per-gene nutrigenomic prescription above.'),
        'diet_impact': "INCREASE: " + "; ".join(r['nutrient'] for r in kb.get('increase', [])[:3]) + ("  |  AVOID: " + "; ".join(r['nutrient'] for r in kb.get('avoid', [])[:3]) if kb.get('avoid') else "")
    }

PATHWAY_KB_EXTRAS = {
    'HBB': {'pathway': 'Beta-Globin Chain Synthesis', 'mechanism': 'HBB encodes the beta-globin chain. Variants cause beta-thalassemia (reduced chains) or sickle cell disease. Even carrier status causes mild microcytic anemia.', 'anemia_link': 'HBB variants -> reduced Hb assembly -> microcytic hypochromic anemia.', 'diet_impact': 'Monitor ferritin before supplementing iron — thalassemia carriers risk iron overload. Folate and B12 are safe and important.'},
    'HBA1': {'pathway': 'Alpha-Globin Chain Synthesis', 'mechanism': 'HBA1 encodes alpha-globin. Alpha-thalassemia trait is very common in South/Southeast Asian populations. Causes microcytic hypochromic anemia.', 'anemia_link': 'Alpha-thalassemia -> reduced alpha-chains -> Hb instability.', 'diet_impact': 'Avoid unsupervised iron supplementation. High folate important for RBC regeneration.'},
    'PKLR': {'pathway': 'RBC Glycolysis — Pyruvate Kinase', 'mechanism': 'PKLR deficiency is the most common glycolytic enzyme defect in RBCs — causes chronic haemolytic anemia due to ATP depletion.', 'anemia_link': 'PKLR variant -> pyruvate kinase deficiency -> RBC ATP depletion -> haemolysis.', 'diet_impact': 'Folate critical for RBC regeneration. Antioxidant-rich foods reduce oxidative burden on low-ATP RBCs.'},
    'ANK1': {'pathway': 'RBC Spectrin-Ankyrin Cytoskeleton', 'mechanism': 'ANK1 (ankyrin) anchors spectrin to the RBC membrane. Variants -> hereditary spherocytosis -> fragile RBCs destroyed in spleen.', 'anemia_link': 'ANK1 variants -> spherocytosis -> haemolysis.', 'diet_impact': 'Folate for RBC regeneration. Omega-3 for membrane health.'},
    'ERFE': {'pathway': 'Erythroferrone — Emergency Iron Mobilisation', 'mechanism': 'ERFE suppresses hepcidin during acute iron demand. Low-function ERFE -> slow iron mobilisation after blood loss or menstruation.', 'anemia_link': 'ERFE variants -> delayed iron mobilisation after blood loss -> slow recovery.', 'diet_impact': 'Increase iron foods especially during and after menstruation. Always pair with Vitamin C.'},
    'KLF1': {'pathway': 'Erythroid Krüppel-Like Factor 1', 'mechanism': 'KLF1 regulates globin gene expression and RBC maturation genes. Variants impair red cell maturation kinetics.', 'anemia_link': 'KLF1 variants -> suboptimal erythroid maturation -> lower Hb per cell.', 'diet_impact': 'Ensure iron, folate, B12 are all optimal for complete erythropoiesis.'},
    'BMP6': {'pathway': 'BMP6-Hepcidin Signalling', 'mechanism': 'BMP6 drives hepcidin production in the liver in response to iron stores. Loss-of-function -> low hepcidin -> iron overload risk.', 'anemia_link': 'BMP6 loss-of-function -> iron overload, not deficiency.', 'diet_impact': 'Limit excess haem iron; get ferritin checked annually. Do not self-supplement iron.'},
    'WNT7B': {'pathway': 'WNT Signalling — Haematopoietic Niche', 'mechanism': 'WNT7B maintains bone marrow haematopoietic niche. Anti-inflammatory diet supports niche integrity.', 'anemia_link': 'WNT7B variants may reduce haematopoietic progenitor output.', 'diet_impact': 'Anti-inflammatory diet: omega-3, turmeric, ginger, green tea between meals.'},
    'FAM222B': {'pathway': 'Novel Erythroid Factor (Under Investigation)', 'mechanism': 'FAM222B reaches genome-wide significance for Hb — mechanism under active research.', 'anemia_link': 'GWAS-confirmed association with Hb levels.', 'diet_impact': 'Maintain optimal iron, folate, B12, Vitamin C as precaution.'},
}
for k, v in PATHWAY_KB_EXTRAS.items():
    if k not in PATHWAY_KB:
        PATHWAY_KB[k] = v

FOOD_DB = {
    'HIGH_IRON': [
        ('Spinach (cooked)', '1 cup / 180g', 6.4, 'Eat MORE — pair with lemon juice to boost absorption'),
        ('Lentils / Dal (cooked)', '1 cup / 200g', 6.6, 'Eat MORE — staple protein + iron; soak before cooking'),
        ('Kidney beans / Rajma', '1 cup / 177g', 5.2, 'Eat MORE — soak overnight to reduce phytates'),
        ('Fortified breakfast cereal', '1 serving / 30g', 8.0, 'Eat MORE — choose brands with ≥20% DV iron on label'),
        ('Chickpeas / Chana', '1 cup / 164g', 4.7, 'Eat MORE — use in curries, salads, or roast as snack'),
        ('Tofu (firm)', '½ cup / 126g', 3.4, 'Eat MORE — good plant iron; stir-fry or add to curry'),
        ('Pumpkin seeds', '30g handful', 2.5, 'Eat MORE — snack every day between meals'),
        ('Quinoa (cooked)', '1 cup / 185g', 2.8, 'Eat MORE — complete protein + iron; replace white rice'),
        ('Jaggery / Gur', '1 tbsp / 20g', 1.6, 'Eat MORE — replace refined sugar in chai and desserts'),
        ('Lean red meat (beef/lamb)', '85g / 3 oz', 2.5, 'Eat MORE — haem iron, best absorbed form; 3x per week'),
    ],
    'MODERATE_IRON': [
        ('Eggs (whole)', '2 large', 1.8, 'Eat MORE — yolk contains most iron; boiled or poached'),
        ('Oats (cooked)', '1 cup / 234g', 2.1, 'Eat MORE — iron breakfast; add nuts and seeds'),
        ('Soybeans (cooked)', '½ cup / 86g', 4.4, 'Eat MORE — surprisingly high iron for a moderate food'),
        ('Potato (baked with skin)', '1 medium', 1.9, 'Eat MORE — keep the skin on for maximum iron'),
        ('Broccoli (cooked)', '1 cup / 156g', 1.0, 'Eat MORE — also rich in Vitamin C, boosts absorption'),
        ('Whole wheat bread', '2 slices / 56g', 1.7, 'Eat MORE — choose 100% whole wheat only'),
        ('Brown rice (cooked)', '1 cup / 195g', 1.0, 'Eat MORE — better than white rice'),
        ('Nuts: almonds, cashews', '30g handful', 1.1, 'Eat MORE — iron-rich snack; keep in bag/at desk'),
    ],
    'MAINTENANCE': [
        ('Mixed seasonal fruit bowl', '1 cup', 0.5, 'Maintain daily variety for micronutrients'),
        ('Green leafy salad', '2 cups', 1.0, 'Maintain at least one salad per day'),
        ('Whole grain roti / chapati', '2 medium', 1.8, 'Maintain as daily carb base'),
        ('Dairy: milk or yogurt', '1 cup', 0.1, 'Maintain — take separately from iron meals'),
    ],
    'VITAMIN_C': [
        ('Amla / Indian gooseberry', '2 small / 50g', 'Highest Vit C food — 700% DV; eat with every iron meal'),
        ('Lemon juice (fresh)', 'Squeeze ½ lemon', 'Squeeze directly over dal, spinach, or salad at table'),
        ('Oranges', '1 medium', 'Drink 1 glass orange juice with iron-rich meals'),
        ('Guava', '1 medium', 'Very high Vit C — eat as snack after iron meals'),
        ('Red/yellow bell pepper', '½ cup raw', 'Chop raw into salads alongside iron foods'),
        ('Tomatoes', '1 medium', 'Add to every meal — raw or cooked both effective'),
    ],
    'OMEGA3': [
        ('Flaxseeds / Alsi', '1 tbsp ground', 'Grind and add to roti dough or sprinkle on dal'),
        ('Walnuts', '4–5 whole / 30g', 'Anti-inflammatory; eat as daily snack'),
        ('Fatty fish (salmon/mackerel)', '100g 2x/week', 'Best omega-3 source; reduces inflammatory hepcidin'),
        ('Chia seeds', '1 tbsp / 12g', 'Add to water, smoothie, or yogurt'),
    ],
    'VITAMIN_B6': [
        ('Banana', '1 medium', 'Good B6; easy daily snack'),
        ('Chicken breast', '85g / 3 oz', 'Excellent B6 + haem iron combination'),
        ('Potato (boiled with skin)', '1 medium', 'Good B6; keep skin on'),
        ('Pistachios', '30g / 23 nuts', 'B6-rich nut; healthy snack'),
        ('Fish (tuna/salmon)', '85g / 3 oz', 'High B6 + omega-3 + haem iron'),
    ],
    'VITAMIN_B12': [
        ('Eggs', '2 large daily', 'Good B12; veg-friendly animal source'),
        ('Dairy: milk, paneer, curd', '1 cup / 150g', 'Easy B12 source for vegetarians'),
        ('Fish / chicken', '85g 3x/week', 'Best B12 food source'),
        ('Fortified plant milk', '1 cup', 'Soy/oat/almond milk with B12 added'),
        ('Nutritional yeast', '1 tbsp / 10g', 'Sprinkle on food; high B12 for vegans'),
    ],
    'FOLATE': [
        ('Spinach / Palak', '1 cup cooked', 'Highest folate among common vegetables'),
        ('Lentils (any dal)', '1 cup cooked', 'Excellent folate + iron double benefit'),
        ('Fenugreek / Methi leaves', '1 cup cooked', 'Very high folate; use in sabzi or paratha'),
        ('Asparagus', '½ cup / 90g', 'Very high folate; roast or steam'),
        ('Fortified cereal', '1 serving', 'Check label for 25–50% DV folate'),
        ('Citrus fruits', '1 medium daily', 'Folate + Vit C double benefit'),
    ],
    'ANTIOXIDANT': [
        ('Blueberries / Jamun', '½ cup / 75g', 'High antioxidant; protects RBCs from oxidative damage'),
        ('Turmeric milk (haldi doodh)', '1 cup', 'Anti-inflammatory; reduce hepcidin elevation'),
        ('Green tea', '1 cup (not with meals)', 'Antioxidant; wait 1h after iron meals'),
        ('Beetroot', '1 medium / 150g', 'Iron + antioxidants; juice or roasted'),
        ('Pomegranate', '½ cup seeds', 'Antioxidant + some iron; great in salads'),
    ],
    'AVOID_STANDARD': [
        ('Tea / Coffee', 'Wait ≥1 hour after iron-rich meals', 'Tannins block iron absorption by up to 60%'),
        ('Calcium supplements', 'Take 2 hours apart from iron meals', 'Calcium directly competes with iron for intestinal absorption'),
        ('Unsoaked raw legumes', 'Always soak 8h + cook thoroughly', 'Phytic acid in unsoaked legumes reduces iron bioavailability'),
        ('Antacids / PPIs', 'Take 2 hours away from iron-rich meals', 'Reduce stomach acid needed to convert Fe³⁺ to absorbable Fe²⁺'),
    ],
    'AVOID_OXIDANT': [
        ('Fava beans / Broad beans', 'AVOID completely if G6PD deficiency suspected', 'Triggers acute haemolytic crisis in G6PD-deficient individuals'),
        ('High-dose Vitamin C supplements', 'Limit to food sources only', 'Pro-oxidant at high doses in G6PD deficiency'),
        ('Naphthalene (mothballs)', 'Avoid household exposure', 'Triggers haemolysis in G6PD deficiency'),
    ],
    'AVOID_IRON_OVERLOAD': [
        ('Excess red meat', 'Limit to 2x per week maximum', 'Haem iron overload risk when ferroportin/BMP6 pathway affected'),
        ('Iron supplements (OTC)', 'Do NOT self-supplement without blood test', 'Iron overload is dangerous; get serum ferritin tested first'),
        ('Vitamin C with every meal', 'Use strategically, not constantly', 'Excess Vit C can enhance iron overload in susceptible individuals'),
    ],
}

MEAL_PLANS = {
    'SEVERE': {
        'title': '🔴 SEVERE Anemia — Intensive Iron Protocol',
        'breakfast': 'Fortified cereal (1 serving) + 2 boiled eggs + amla juice (100ml) + 1 orange',
        'mid_morning': 'Pumpkin seeds (30g) + guava (1 medium)',
        'lunch': 'Lentil dal (1 cup) + palak/spinach sabzi (1 cup) + 2 whole-wheat rotis + lemon squeezed over dal',
        'snack': 'Jaggery-peanut chikki (30g) + 1 glass orange juice',
        'dinner': 'Rajma curry (1 cup) + brown rice (1 cup) + broccoli stir-fry + tomato salad',
        'note': '⚠️ Seek immediate medical attention. Diet supports — does not replace — clinical treatment.',
    },
    'HIGH': {
        'title': '🟠 Low Hemoglobin — High Iron Diet Required',
        'breakfast': 'Oats (1 cup) cooked with jaggery + 2 boiled eggs + 1 glass orange juice',
        'mid_morning': 'Amla (2 pieces) OR amla candy + mixed nuts (30g)',
        'lunch': 'Dal (1 cup) + palak paneer / spinach curry (1 cup) + 2 rotis + squeeze lemon',
        'snack': 'Roasted chickpeas (½ cup) + lemon water',
        'dinner': 'Rajma / kidney bean curry (1 cup) + quinoa (1 cup) + stir-fried broccoli',
        'note': '🔁 Repeat iron-rich meals every day. ALWAYS pair every iron food with a Vitamin C source.',
    },
    'MODERATE': {
        'title': '🟡 Borderline Hemoglobin — Moderate Iron Diet',
        'breakfast': '2 whole-wheat toast + 1 egg + tomato + 1 glass guava juice',
        'mid_morning': 'Banana + 10–12 almonds',
        'lunch': 'Chana dal (½ cup) + vegetable sabzi + 2 rotis + cucumber-lemon salad',
        'snack': 'Pumpkin seeds (20g) + orange or guava',
        'dinner': 'Brown rice + tofu / chicken curry + bell pepper stir-fry',
        'note': '📅 Check hemoglobin every 3 months. Maintain consistent iron-rich choices.',
    },
    'NORMAL': {
        'title': '🟢 Normal Hemoglobin — Maintenance Diet',
        'breakfast': 'Whole-grain cereal or oats + milk + seasonal fruit',
        'mid_morning': 'Fruit or nuts',
        'lunch': 'Balanced dal-rice or roti-sabzi with salad',
        'snack': 'Yogurt + fruit',
        'dinner': 'Mixed diet with greens, legumes, and grains',
        'note': '✅ Maintain variety and balance. Continue regular health check-ups.',
    }
}

PATHWAY_MEAL_OVERRIDES = {
    'hepcidin': {
        'breakfast': 'Fatty fish (50g) OR 2 boiled eggs + amla juice (100ml) + 1 orange + oats with jaggery [Vit C + heme iron critical for hepcidin block]',
        'lunch': 'Lean red meat curry (85g) OR lentil dal (1 cup) + palak sabzi + 2 rotis + squeeze lemon DIRECTLY over all iron foods',
        'dinner': 'Fish curry (100g) + brown rice + broccoli + turmeric milk before bed [omega-3 reduces IL-6-driven hepcidin]',
        'note': '⚠️ HEPCIDIN-AXIS GENE ACTIVE: Oral iron absorption may be systemically limited. EVERY iron meal must have Vitamin C. Prioritise heme iron over plant iron. Clinical hepcidin measurement recommended.',
    },
    'absorption': {
        'breakfast': '2 boiled eggs + fortified cereal + FRESH amla juice (not from concentrate) + orange wedges [DcytB/DMT1 variant — Vit C is non-negotiable at this meal]',
        'lunch': 'Lentil dal (1 cup, well-soaked 8h) + palak sabzi + lemon squeezed over dal + tomato salad [fermentation/soaking reduces phytates for impaired DMT1]',
        'dinner': 'Fish OR chicken (85g) + brown rice + broccoli stir-fry with bell pepper [heme iron bypasses defective DMT1; plant Vit C boosts remaining non-heme absorption]',
        'note': '⚠️ ABSORPTION GENE ACTIVE: Vitamin C is NOT optional — it must accompany EVERY iron-containing food. Soak all legumes 8+ hours. Prioritise heme iron sources where possible.',
    },
    'rbc_membrane': {
        'breakfast': 'Oats + 1 tbsp ground flaxseed + 2 eggs + orange juice [omega-3 for membrane; Vit C for iron]',
        'lunch': 'Salmon/mackerel (100g) 2×/week OR tofu + quinoa + spinach salad with olive oil dressing [pre-formed EPA/DHA for membrane lipids]',
        'dinner': 'Dal + brown rice + broccoli + 4–5 walnuts as side + 2.5L water throughout day [hydration critical for dehydrated-RBC pathways]',
        'note': '🟠 RBC MEMBRANE GENE ACTIVE: Omega-3 fatty acids (especially pre-formed EPA/DHA from fish) are the primary dietary target. Avoid ALL trans-fats. Stay well-hydrated.',
    },
    'haem_synthesis': {
        'breakfast': '2 eggs + banana + whole-wheat toast + orange juice [eggs for iron; banana for B6; OJ for Vit C]',
        'lunch': 'Chicken (85g) OR fish + brown rice + palak sabzi [B6 + iron + folate triple benefit]',
        'dinner': 'Dal + potato (with skin, for B6) + 2 rotis + squeeze lemon [B6 from potato; non-heme iron enhanced by lemon]',
        'note': '⚠️ HAEM SYNTHESIS GENE ACTIVE: Vitamin B6 is as important as iron. Include B6-rich food (chicken/fish/banana/potato) at EVERY meal. B6 supplement (25–50mg/day) recommended clinically.',
    },
    'folate_b12': {
        'breakfast': 'Fortified cereal (check for B12 on label) + milk + citrus fruit + methi leaves (folate) in paratha OR scrambled eggs',
        'lunch': 'Spinach/methi dal (dark leafy greens = folate) + eggs OR chicken (B12) + 2 rotis + lemon salad',
        'dinner': 'Fish OR paneer curry + brown rice + dark leafy green sabzi (palak/methi) + 1 glass fortified milk',
        'note': '⚠️ FOLATE/B12 PATHWAY GENE ACTIVE: Methylfolate supplement (5-MTHF form, NOT standard folic acid) strongly recommended. B12 supplement for vegetarians. Check plasma homocysteine.',
    },
    'rbc_energy': {
        'breakfast': 'Lentil soup (folate) + 2 eggs + orange juice [folate for purine synthesis; Vit C for iron]',
        'lunch': 'Chickpea/rajma curry (folate + iron) + brown rice + raw tomato salad [legumes provide purine precursors and folate]',
        'dinner': 'Dal + spinach sabzi + 2 rotis — drink 500ml water with this meal [folate; hydration critical for RBC ATP support]',
        'note': '🟠 RBC ENERGY GENE ACTIVE: Folate is the primary dietary target. Hydration is critical — ATP-depleted RBCs worsen with dehydration. Avoid prolonged fasting.',
    },
    'oxidative_protection': {
        'breakfast': 'Oats + blueberries/jamun (½ cup) + 2 eggs + orange — NO fava beans or broad beans EVER',
        'lunch': 'Lentil dal + brown rice + vegetables (avoid fava beans) + turmeric in all curries [curcumin = antioxidant for G6PD-deficient RBCs]',
        'dinner': 'Chicken/fish curry + brown rice + jamun or pomegranate seeds as dessert [antioxidants protect RBCs from haemolytic triggers]',
        'note': '🚨 G6PD PATHWAY GENE ACTIVE: FAVA BEANS (BROAD BEANS) ARE ABSOLUTELY PROHIBITED — even once. No high-dose Vit C supplements. Eat antioxidant-rich foods (jamun, turmeric, blueberries) daily.',
    },
    'iron_sensing': {
        'breakfast': 'Green tea (polyphenols) with breakfast — NOT with iron meal [polyphenols limit excess absorption if iron overload risk]',
        'lunch': 'Balanced dal-vegetable meal — NO iron supplements until ferritin tested [may be in overload state, not deficiency]',
        'dinner': 'Cruciferous vegetables (broccoli, cauliflower) + legumes + whole grains',
        'note': '⚠️ IRON SENSING GENE ACTIVE (HFE/TFR2): DO NOT add iron supplements without serum ferritin + transferrin saturation test. Risk of hemochromatosis. Clinical genetics referral recommended.',
    },
}

PATHWAY_AXES = ['hepcidin', 'absorption', 'transferrin', 'erythropoiesis',
                'folate_b12', 'rbc_membrane', 'rbc_energy', 'haem_synthesis',
                'iron_sensing', 'oxidative_protection']

# Preloaded Patients data
PRELOADED_PATIENTS = {
    "patient_1": {
        "label": "Severe Anemia - Hepcidin Axis (TMPRSS6 Mutant)",
        "phenotype": {
            "RIAGENDR": 1, # Female
            "RIDAGEYR": 28,
            "INDFMPIR": 1.2,
            "BMXWT": 52.0,
            "BMXHT": 158.0,
            "BMXBMI": 20.8,
            "BMXWAIST": 70.0,
            "DR1TIRON": 5.2,
            "DR1TVC": 15.0,
            "DR1TPROT": 42.0,
            "DR1TCALC": 600.0,
            "DR1TKCAL": 1600.0
        },
        "snps": {
            "rs2251655": 2 # TMPRSS6 Homozygous
        },
        "diet_pref": "Vegetarian",
        "cuisine_pref": "Indian"
    },
    "patient_2": {
        "label": "High Risk - RBC Lipid Membrane Axis (ACSL3P1 Mutant)",
        "phenotype": {
            "RIAGENDR": 1, # Female
            "RIDAGEYR": 34,
            "INDFMPIR": 2.4,
            "BMXWT": 64.0,
            "BMXHT": 165.0,
            "BMXBMI": 23.5,
            "BMXWAIST": 78.0,
            "DR1TIRON": 10.5,
            "DR1TVC": 10.0, # extremely low Vit C
            "DR1TPROT": 50.0,
            "DR1TCALC": 900.0,
            "DR1TKCAL": 1800.0
        },
        "snps": {
            "rs6762719": 2 # ACSL3P1 Homozygous
        },
        "diet_pref": "Any",
        "cuisine_pref": "Western"
    },
    "patient_3": {
        "label": "Moderate Risk - Dehydrated RBCs (PIEZO1/ATP2B4 Variants)",
        "phenotype": {
            "RIAGENDR": 0, # Male
            "RIDAGEYR": 22,
            "INDFMPIR": 3.0,
            "BMXWT": 72.0,
            "BMXHT": 178.0,
            "BMXBMI": 22.7,
            "BMXWAIST": 82.0,
            "DR1TIRON": 14.0,
            "DR1TVC": 60.0,
            "DR1TPROT": 65.0,
            "DR1TCALC": 1000.0,
            "DR1TKCAL": 2400.0
        },
        "snps": {
            "rs551118": 1, # PIEZO1 heterozygous
            "rs7546390": 2  # ATP2B4 homozygous
        },
        "diet_pref": "Any",
        "cuisine_pref": "Mediterranean"
    },
    "patient_4": {
        "label": "Healthy Baseline - Low Genetic & Dietary Risk",
        "phenotype": {
            "RIAGENDR": 0, # Male
            "RIDAGEYR": 45,
            "INDFMPIR": 4.5,
            "BMXWT": 80.0,
            "BMXHT": 180.0,
            "BMXBMI": 24.7,
            "BMXWAIST": 88.0,
            "DR1TIRON": 18.0,
            "DR1TVC": 120.0,
            "DR1TPROT": 75.0,
            "DR1TCALC": 1100.0,
            "DR1TKCAL": 2200.0
        },
        "snps": {},
        "diet_pref": "Any",
        "cuisine_pref": "Western"
    }
}

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def get_kb_entry(gene):
    if gene in GENE_NUTRITION_KB:
        return GENE_NUTRITION_KB[gene]
    for k in GENE_NUTRITION_KB:
        if gene.startswith(k) or k.startswith(gene):
            return GENE_NUTRITION_KB[k]
    return None

def compute_pathway_burden_scores(active_genes_list):
    raw_scores = {axis: 0.0 for axis in PATHWAY_AXES}
    for gene_entry in active_genes_list:
        gene = gene_entry['gene']
        dosage = gene_entry['dosage']
        beta = abs(gene_entry['beta'])
        
        kb_key = None
        if gene in GENE_NUTRITION_KB:
            kb_key = gene
        else:
            for k in GENE_NUTRITION_KB:
                if gene.startswith(k) or k.startswith(gene):
                    kb_key = k
                    break
        if kb_key:
            axis = GENE_NUTRITION_KB[kb_key].get('pathway_class', None)
            if axis and axis in raw_scores:
                raw_scores[axis] += beta * dosage

    max_raw = max(raw_scores.values()) if any(v > 0 for v in raw_scores.values()) else 1.0
    normaliser = max(max_raw, 1.0)

    burden_scores = {
        axis: round(min(100.0, (score / normaliser) * 100), 1)
        for axis, score in raw_scores.items()
    }
    ranked_axes = sorted(burden_scores.items(), key=lambda x: x[1], reverse=True)
    return burden_scores, ranked_axes

def build_nutrient_requirement_profile(active_genes_list, burden_scores, ranked_axes):
    nutrient_scores = {}
    avoidance_scores = {}
    clinical_flags = []
    evidence_weight = {'Strong': 3, 'Moderate': 2, 'Weak': 1}

    for gene_entry in active_genes_list:
        gene = gene_entry['gene']
        kb = get_kb_entry(gene)
        if not kb:
            continue
        if kb.get('clinical_flag', False):
            clinical_flags.append({
                'gene': gene,
                'note': kb['clinical_note'],
            })
        for rec in kb.get('increase', []):
            key = rec['nutrient'].lower().strip()
            ew = evidence_weight.get(rec['evidence'], 1)
            if key not in nutrient_scores:
                nutrient_scores[key] = {
                    'nutrient': rec['nutrient'],
                    'evidence': rec['evidence'],
                    'weight': ew,
                    'rationale': rec['rationale'],
                    'genes': [gene],
                    'evidence_lvl': ew,
                }
            else:
                nutrient_scores[key]['weight'] += ew
                nutrient_scores[key]['evidence_lvl'] = max(nutrient_scores[key]['evidence_lvl'], ew)
                if gene not in nutrient_scores[key]['genes']:
                    nutrient_scores[key]['genes'].append(gene)

        for rec in kb.get('avoid', []):
            key = rec['nutrient'].lower().strip()
            ew = evidence_weight.get(rec['evidence'], 1)
            if key not in avoidance_scores:
                avoidance_scores[key] = {
                    'nutrient': rec['nutrient'],
                    'evidence': rec['evidence'],
                    'weight': ew,
                    'rationale': rec['rationale'],
                    'genes': [gene],
                    'evidence_lvl': ew,
                }
            else:
                avoidance_scores[key]['weight'] += ew
                avoidance_scores[key]['evidence_lvl'] = max(avoidance_scores[key]['evidence_lvl'], ew)
                if gene not in avoidance_scores[key]['genes']:
                    avoidance_scores[key]['genes'].append(gene)

    conflicts = []
    for key in list(nutrient_scores.keys()):
        if key in avoidance_scores:
            inc_genes = nutrient_scores[key]['genes']
            avd_genes = avoidance_scores[key]['genes']
            conflicts.append({
                'nutrient': nutrient_scores[key]['nutrient'],
                'increase_via': inc_genes,
                'avoid_via': avd_genes,
                'resolution': f"CONFLICT: {inc_genes} suggests INCREASE; {avd_genes} suggests AVOID. CLINICAL TEST REQUIRED (e.g. serum ferritin/transferrin saturation) to determine which direction applies to this patient.",
            })

    priority_nutrients = sorted(nutrient_scores.values(), key=lambda x: x['weight'], reverse=True)
    priority_avoidances = sorted(avoidance_scores.values(), key=lambda x: x['weight'], reverse=True)

    return priority_nutrients, priority_avoidances, conflicts, clinical_flags

def select_meal_override(burden_scores, ranked_axes, active_genes_list):
    for axis, score in ranked_axes:
        if score > 20 and axis in PATHWAY_MEAL_OVERRIDES:
            driving_genes = [
                g['gene'] for g in active_genes_list
                if get_kb_entry(g['gene']) and get_kb_entry(g['gene']).get('pathway_class') == axis
            ]
            reason = f"{axis.upper()} axis (Score: {score}/100) — driven by {', '.join(driving_genes[:3])}"
            return PATHWAY_MEAL_OVERRIDES[axis], reason
    return None, "none (standard Hb-based plan — no dominant pathway burden detected)"

def parse_vcf_bytes(vcf_bytes):
    try:
        content = gzip.decompress(vcf_bytes).decode('utf-8', errors='ignore')
    except Exception:
        content = vcf_bytes.decode('utf-8', errors='ignore')

    rows = []
    for line in content.split('\n'):
        if line.startswith('#') or not line.strip():
            continue
        parts = line.strip().split('\t')
        if len(parts) < 5:
            continue
        chrom, pos, rsid, ref, alt = parts[0], parts[1], parts[2], parts[3], parts[4]
        format_field = parts[8] if len(parts) > 8 else ''
        sample_field = parts[9] if len(parts) > 9 else ''
        gt = '.'
        if 'GT' in format_field:
            fmt_keys = format_field.split(':')
            smp_vals = sample_field.split(':')
            if 'GT' in fmt_keys:
                gt = smp_vals[fmt_keys.index('GT')]
        rows.append({'RSID': rsid, 'CHROM': chrom, 'POS': pos, 'REF': ref, 'ALT': alt, 'GT': gt})
    return pd.DataFrame(rows)

def gt_to_dosage(gt):
    if pd.isna(gt) or gt in ('.', './.', '.|.'):
        return np.nan
    gt_clean = gt.replace('|', '/').split(':')[0]
    alleles = gt_clean.split('/')
    try:
        return sum(int(a) for a in alleles if a != '.')
    except ValueError:
        return np.nan

# -----------------------------------------------------------------------------
# PIPELINE LOADING & MODEL TRAINING
# -----------------------------------------------------------------------------
def load_datasets_and_train():
    if not os.path.exists(GWAS_PATH) or not os.path.exists(NHANES_PATH):
        raise FileNotFoundError("Missing GWAS or NHANES CSV file in /Users/yashna/IDPMODEL/")

    # 1. Load GWAS
    gwas = pd.read_csv(GWAS_PATH, sep=None, engine='python')
    gwas.columns = gwas.columns.str.strip()
    for col in ['P_value', 'Beta', 'RAF']:
        gwas[col] = pd.to_numeric(gwas[col].astype(str).str.strip(), errors='coerce')
    gwas = gwas.dropna(subset=['SNPS', 'Genes', 'P_value', 'Beta', 'RAF'])
    gwas = gwas[(gwas['P_value'] > 0) & (gwas['P_value'] <= 1) &
                (gwas['RAF'].between(0.01, 0.99)) &
                (gwas['Beta'].abs() > 0.0001)]
    gwas = gwas.sort_values('P_value').drop_duplicates('SNPS').head(400).reset_index(drop=True)
    gwas['Genes'] = gwas['Genes'].astype(str).str.strip().replace(['nan','None',''], 'Intergenic')

    # Get GWAS significant SNPs
    gwas_sig = gwas[gwas['P_value'] < 5e-8].copy()
    if len(gwas_sig) < 10:
        gwas_sig = gwas.head(50)

    snp_gene_map = gwas_sig[['SNPS', 'Genes']].copy()
    snp_ids = gwas_sig['SNPS'].tolist()
    rafs = gwas_sig['RAF'].tolist()
    betas = gwas_sig['Beta'].tolist()

    # 2. Load NHANES
    nhanes = pd.read_csv(NHANES_PATH)
    nhanes.columns = nhanes.columns.str.strip()
    nhanes = nhanes.dropna(subset=['LBXHGB'])
    
    # Fill NHANES medians
    nhanes_median = nhanes.median(numeric_only=True).to_dict()
    for col in nhanes.select_dtypes(include=[np.number]).columns:
        nhanes[col] = nhanes[col].fillna(nhanes_median[col])

    if nhanes['RIAGENDR'].isin([1, 2]).all():
        nhanes['RIAGENDR'] = nhanes['RIAGENDR'].map({1: 0, 2: 1}) # 0 = Male, 1 = Female

    # Simulate genotypes
    n_patients = len(nhanes)
    genotype_matrix = np.zeros((n_patients, len(snp_ids)), dtype=np.float32)
    for j, (snp, p, beta) in enumerate(zip(snp_ids, rafs, betas)):
        p = float(np.clip(p, 0.001, 0.999))
        probs = [(1-p)**2, 2*p*(1-p), p**2]
        genotype_matrix[:, j] = np.random.choice([0, 1, 2], size=n_patients, p=probs)

    geno_df = pd.DataFrame(genotype_matrix, columns=snp_ids)
    prs_scores = genotype_matrix @ np.array(betas, dtype=np.float32)

    nhanes['PRS'] = prs_scores
    p25 = np.percentile(prs_scores, 25)
    p75 = np.percentile(prs_scores, 75)

    def prs_cat_func(s):
        if s < p25: return 'Low'
        elif s <= p75: return 'Moderate'
        else: return 'High'
    nhanes['PRS_category'] = nhanes['PRS'].apply(prs_cat_func)

    # Encode categorical columns
    cat_cols = nhanes.select_dtypes(include=['object']).columns.tolist()
    cat_cols = [c for c in cat_cols if c != 'PRS_category']
    le_enc = {}
    for col in cat_cols:
        le = LabelEncoder()
        nhanes[col] = le.fit_transform(nhanes[col].astype(str))
        le_enc[col] = le

    prs_le = LabelEncoder()
    nhanes['PRS_category_enc'] = prs_le.fit_transform(nhanes['PRS_category'])

    # Build final ML ready dataframe
    geno_df.index = nhanes.index
    df = pd.concat([nhanes, geno_df], axis=1)

    # Anemia labels (WHO thresholds)
    def create_label(row):
        return 1 if (row['RIAGENDR'] == 1 and row['LBXHGB'] < 12.0) \
                 or (row['RIAGENDR'] == 0 and row['LBXHGB'] < 13.0) else 0
    df['iron_deficient'] = df.apply(create_label, axis=1)
    hb_values = df['LBXHGB'].copy()

    # Drop leakage columns
    CBC_LEAKAGE_COLS = [
        'LBXHGB', 'LBXHCT', 'LBXRBCSI', 'LBXMCVSI', 'LBXMCHSI', 'LBXRDW', 
        'iron_deficient', 'PRS_category'
    ]
    drop_cols = [c for c in CBC_LEAKAGE_COLS if c in df.columns]
    
    # Ensure bool columns to int
    bool_cols = df.select_dtypes(include='bool').columns.tolist()
    df[bool_cols] = df[bool_cols].astype(int)

    X = df.drop(columns=drop_cols)
    y_class = df['iron_deficient']
    y_reg = hb_values.values

    # Drop any leftover non-numeric columns
    X = X.apply(pd.to_numeric, errors='coerce').fillna(X.median(numeric_only=True))
    feature_names = X.columns.tolist()

    # Save mapping for UniProt info
    snp_gene_map['gene_clean'] = snp_gene_map['Genes'].apply(
        lambda x: str(x).split(',')[0].strip().split('-')[0].strip()
    )

    # -------------------------------------------------------------------------
    # Train Models (Classifier & Regressor)
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    clf = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=4,
        min_child_weight=5,
        subsample=0.75,
        colsample_bytree=0.6,
        reg_alpha=0.5,
        reg_lambda=2.0,
        eval_metric='logloss',
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    clf.fit(
        X_train_sm, y_train_sm,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Evaluate Classifier
    y_pred_class = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    train_prob = clf.predict_proba(X_train_sm)[:, 1]
    clf_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred_class)),
        "precision": float(precision_score(y_test, y_pred_class)),
        "recall": float(recall_score(y_test, y_pred_class)),
        "f1": float(f1_score(y_test, y_pred_class)),
        "test_auc": float(roc_auc_score(y_test, y_prob)),
        "train_auc": float(roc_auc_score(y_train_sm, train_prob)),
        "best_iteration": int(clf.best_iteration)
    }

    # Train Regressor
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    reg = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=4,
        min_child_weight=5,
        subsample=0.75,
        colsample_bytree=0.6,
        reg_alpha=0.5,
        reg_lambda=2.0,
        eval_metric='rmse',
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    reg.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        verbose=False
    )

    # Evaluate Regressor
    y_pred_reg = reg.predict(X_te)
    train_pred_reg = reg.predict(X_tr)
    reg_metrics = {
        "mae": float(mean_absolute_error(y_te, y_pred_reg)),
        "rmse": float(np.sqrt(mean_squared_error(y_te, y_pred_reg))),
        "test_r2": float(r2_score(y_te, y_pred_reg)),
        "train_r2": float(r2_score(y_tr, train_pred_reg)),
        "best_iteration": int(reg.best_iteration)
    }

    # Save to global model store
    MODELS["classifier"] = clf
    MODELS["regressor"] = reg
    MODELS["feature_names"] = feature_names
    MODELS["snp_ids"] = snp_ids
    MODELS["gwas_sig"] = gwas_sig
    MODELS["snp_bio"] = snp_gene_map
    MODELS["nhanes_median"] = nhanes_median
    MODELS["prs_p25"] = float(p25)
    MODELS["prs_p75"] = float(p75)
    MODELS["prs_le"] = prs_le
    MODELS["is_trained"] = True

    return {
        "status": "success",
        "classifier": clf_metrics,
        "regressor": reg_metrics,
        "n_features": len(feature_names),
        "n_snps": len(snp_ids)
    }

# -----------------------------------------------------------------------------
# STARTUP EVENT
# -----------------------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    try:
        load_datasets_and_train()
        print("✅ Models trained and data pipeline successfully loaded on startup.")
    except Exception as e:
        print(f"⚠ Failed to load/train pipeline on startup: {e}")

# -----------------------------------------------------------------------------
# API SCHEMA DEFINITIONS
# -----------------------------------------------------------------------------
class PhenotypeData(BaseModel):
    RIAGENDR: int # 0 = Male, 1 = Female
    RIDAGEYR: int
    INDFMPIR: float
    BMXWT: float
    BMXHT: float
    BMXBMI: float
    BMXWAIST: float
    DR1TIRON: float
    DR1TVC: float
    DR1TPROT: float
    DR1TCALC: float
    DR1TKCAL: float

class PatientInput(BaseModel):
    phenotype: PhenotypeData
    snps: Dict[str, int]
    diet_pref: Optional[str] = "Any"
    cuisine_pref: Optional[str] = "Western"

# -----------------------------------------------------------------------------
# ENDPOINTS
# -----------------------------------------------------------------------------
@app.post("/api/train")
def api_train_model():
    try:
        metrics = load_datasets_and_train()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/api/preloaded-patients")
def get_preloaded_patients():
    return PRELOADED_PATIENTS

@app.post("/api/parse-vcf")
def api_parse_vcf(file: UploadFile = File(...)):
    if not MODELS["is_trained"]:
        raise HTTPException(status_code=500, detail="Model pipeline not trained yet.")
    
    try:
        file_bytes = file.file.read()
        vcf_df = parse_vcf_bytes(file_bytes)
        vcf_df['Dosage'] = vcf_df['GT'].apply(gt_to_dosage)

        # Compute PRS
        gwas_sig = MODELS["gwas_sig"]
        gwas_sig_dict = dict(zip(gwas_sig['SNPS'], gwas_sig['Beta']))
        
        patient_prs = 0.0
        matched_snps = []
        unmatched_snps = []

        vcf_dict = dict(zip(vcf_df['RSID'], vcf_df['Dosage']))

        for snp_id, beta_val in gwas_sig_dict.items():
            dosage = vcf_dict.get(snp_id, np.nan)
            if not np.isnan(dosage) and dosage > 0:
                patient_prs += dosage * beta_val
                matched_snps.append({
                    "snp": snp_id,
                    "dosage": int(dosage),
                    "beta": float(beta_val)
                })
            else:
                unmatched_snps.append(snp_id)

        p25 = MODELS["prs_p25"]
        p75 = MODELS["prs_p75"]
        prs_cat = 'Low' if patient_prs < p25 else 'Moderate' if patient_prs <= p75 else 'High'

        # Map to gene info
        snp_bio = MODELS["snp_bio"]
        active_genes_list = []
        for match in matched_snps:
            snp_id = match["snp"]
            dosage = match["dosage"]
            beta_val = match["beta"]
            # Find gene
            g_rows = snp_bio[snp_bio['SNPS'] == snp_id]
            gene = g_rows['gene_clean'].values[0] if len(g_rows) > 0 else "Unknown"
            active_genes_list.append({
                "gene": gene,
                "snp": snp_id,
                "beta": beta_val,
                "dosage": dosage
            })

        return {
            "prs": float(patient_prs),
            "prs_category": prs_cat,
            "matched_count": len(matched_snps),
            "total_significant_snps": len(gwas_sig_dict),
            "matched_snps": matched_snps,
            "active_genes": active_genes_list
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"VCF parsing failed: {str(e)}")

# -----------------------------------------------------------------------------
# DYNAMIC NUTRIGENOMIC MEAL PLANNER
# -----------------------------------------------------------------------------
DYNAMIC_FOOD_DB = [
    # BREAKFASTS
    {
        "name": "Ragi Malt / Finger Millet Porridge",
        "quantity": "1 cup (cooked)",
        "iron": 3.7, "vit_c": 0.0, "b6": 0.1, "folate": 12.0, "b12": 0.0,
        "veg": True, "vegan": True,
        "cuisines": ["Indian"], "categories": ["breakfast"]
    },
    {
        "name": "Methi (Fenugreek) Paratha",
        "quantity": "2 pieces",
        "iron": 2.8, "vit_c": 5.0, "b6": 0.1, "folate": 45.0, "b12": 0.0,
        "veg": True, "vegan": False,
        "cuisines": ["Indian"], "categories": ["breakfast"]
    },
    {
        "name": "Oats cooked with Jaggery & Chia Seeds",
        "quantity": "1 cup",
        "iron": 3.2, "vit_c": 0.0, "b6": 0.2, "folate": 20.0, "b12": 0.0,
        "veg": True, "vegan": True,
        "cuisines": ["Western"], "categories": ["breakfast"]
    },
    {
        "name": "Fortified Breakfast Cereal with Almond Milk",
        "quantity": "1 serving (30g)",
        "iron": 8.0, "vit_c": 5.0, "b6": 0.5, "folate": 100.0, "b12": 1.5,
        "veg": True, "vegan": True,
        "cuisines": ["Western"], "categories": ["breakfast"]
    },
    {
        "name": "Hummus & Avocado on Sourdough Toast",
        "quantity": "2 slices + 1/2 cup hummus",
        "iron": 3.5, "vit_c": 12.0, "b6": 0.3, "folate": 60.0, "b12": 0.0,
        "veg": True, "vegan": True,
        "cuisines": ["Mediterranean"], "categories": ["breakfast"]
    },
    {
        "name": "Shakshuka (Eggs poached in tomato & bell pepper sauce)",
        "quantity": "2 eggs + 1 cup sauce",
        "iron": 3.2, "vit_c": 35.0, "b6": 0.3, "folate": 50.0, "b12": 1.2,
        "veg": True, "vegan": False,
        "cuisines": ["Mediterranean", "Western"], "categories": ["breakfast"]
    },
    {
        "name": "Greek Yogurt with Blueberries & Honey",
        "quantity": "1 cup yogurt",
        "iron": 0.2, "vit_c": 4.0, "b6": 0.1, "folate": 15.0, "b12": 1.4,
        "veg": True, "vegan": False,
        "cuisines": ["Mediterranean", "Western"], "categories": ["breakfast"]
    },
    {
        "name": "Scrambled Tofu with Turmeric & Spinach on Sourdough",
        "quantity": "1 cup scramble",
        "iron": 4.2, "vit_c": 8.0, "b6": 0.2, "folate": 35.0, "b12": 1.0,
        "veg": True, "vegan": True,
        "cuisines": ["Western", "Mediterranean"], "categories": ["breakfast"]
    },
    
    # MAINS (LUNCH & DINNER)
    {
        "name": "Lentil Dal (Soaked Dal-Fry)",
        "quantity": "1.5 cups (cooked)",
        "iron": 7.8, "vit_c": 2.0, "b6": 0.3, "folate": 180.0, "b12": 0.0,
        "veg": True, "vegan": True,
        "cuisines": ["Indian"], "categories": ["lunch", "dinner"]
    },
    {
        "name": "Cooked Spinach Sabzi (Palak)",
        "quantity": "1 cup cooked",
        "iron": 6.4, "vit_c": 18.0, "b6": 0.2, "folate": 260.0, "b12": 0.0,
        "veg": True, "vegan": True,
        "cuisines": ["Indian"], "categories": ["lunch", "dinner"]
    },
    {
        "name": "Palak Paneer",
        "quantity": "1 cup",
        "iron": 4.8, "vit_c": 10.0, "b6": 0.2, "folate": 120.0, "b12": 0.8,
        "veg": True, "vegan": False,
        "cuisines": ["Indian"], "categories": ["lunch", "dinner"]
    },
    {
        "name": "Chicken Tikka Curry",
        "quantity": "150g",
        "iron": 2.2, "vit_c": 5.0, "b6": 0.8, "folate": 15.0, "b12": 0.6,
        "veg": False, "vegan": False,
        "cuisines": ["Indian"], "categories": ["lunch", "dinner"]
    },
    {
        "name": "Rajma (Kidney Bean) Curry",
        "quantity": "1.5 cups",
        "iron": 6.8, "vit_c": 4.0, "b6": 0.3, "folate": 150.0, "b12": 0.0,
        "veg": True, "vegan": True,
        "cuisines": ["Indian"], "categories": ["lunch", "dinner"]
    },
    {
        "name": "Western Quinoa Salad with Black Beans & Steamed Broccoli",
        "quantity": "1.5 cups",
        "iron": 5.8, "vit_c": 80.0, "b6": 0.4, "folate": 110.0, "b12": 0.0,
        "veg": True, "vegan": True,
        "cuisines": ["Western"], "categories": ["lunch", "dinner"]
    },
    {
        "name": "Tofu & Veggie Stir-fry with Sesame Seeds",
        "quantity": "150g tofu + 1 cup veggies",
        "iron": 4.8, "vit_c": 45.0, "b6": 0.3, "folate": 45.0, "b12": 0.0,
        "veg": True, "vegan": True,
        "cuisines": ["Western"], "categories": ["lunch", "dinner"]
    },
    {
        "name": "Grilled Lean Beef Steak with Roasted Asparagus",
        "quantity": "100g beef + 1 cup asparagus",
        "iron": 4.5, "vit_c": 15.0, "b6": 0.6, "folate": 40.0, "b12": 2.2,
        "veg": False, "vegan": False,
        "cuisines": ["Western"], "categories": ["lunch", "dinner"]
    },
    {
        "name": "Falafel, Hummus, & Tabouli Salad",
        "quantity": "4 pieces + 1 cup salad",
        "iron": 4.2, "vit_c": 30.0, "b6": 0.2, "folate": 95.0, "b12": 0.0,
        "veg": True, "vegan": True,
        "cuisines": ["Mediterranean"], "categories": ["lunch", "dinner"]
    },
    {
        "name": "Tuscan White Bean & Spinach Stew",
        "quantity": "1.5 cups",
        "iron": 6.2, "vit_c": 22.0, "b6": 0.3, "folate": 140.0, "b12": 0.0,
        "veg": True, "vegan": True,
        "cuisines": ["Mediterranean"], "categories": ["lunch", "dinner"]
    },
    {
        "name": "Grilled Salmon with Spinach Salad",
        "quantity": "120g fish + 1.5 cups spinach",
        "iron": 5.6, "vit_c": 14.0, "b6": 0.7, "folate": 90.0, "b12": 4.0,
        "veg": False, "vegan": False,
        "cuisines": ["Mediterranean"], "categories": ["lunch", "dinner"]
    },
    
    # SNACKS
    {
        "name": "Roasted Chickpeas (Chana)",
        "quantity": "1/2 cup (80g)",
        "iron": 2.4, "vit_c": 0.0, "b6": 0.1, "folate": 40.0, "b12": 0.0,
        "veg": True, "vegan": True,
        "cuisines": ["Indian"], "categories": ["snack"]
    },
    {
        "name": "Jaggery-Peanut Chikki",
        "quantity": "30g",
        "iron": 1.2, "vit_c": 0.0, "b6": 0.1, "folate": 15.0, "b12": 0.0,
        "veg": True, "vegan": True,
        "cuisines": ["Indian"], "categories": ["snack"]
    },
    {
        "name": "Pumpkin Seeds (Pepitas)",
        "quantity": "30g handful",
        "iron": 2.5, "vit_c": 0.0, "b6": 0.1, "folate": 15.0, "b12": 0.0,
        "veg": True, "vegan": True,
        "cuisines": ["Western", "Mediterranean"], "categories": ["snack"]
    },
    {
        "name": "Walnuts & Almonds Mix",
        "quantity": "30g handful",
        "iron": 1.1, "vit_c": 0.0, "b6": 0.1, "folate": 10.0, "b12": 0.0,
        "veg": True, "vegan": True,
        "cuisines": ["Western", "Mediterranean"], "categories": ["snack"]
    },

    # BOOSTERS
    {
        "name": "Amla Juice (Indian Gooseberry)",
        "quantity": "1 shot (50ml)",
        "iron": 0.2, "vit_c": 300.0, "b6": 0.0, "folate": 5.0, "b12": 0.0,
        "veg": True, "vegan": True,
        "cuisines": ["Indian"], "categories": ["booster"]
    },
    {
        "name": "Freshly Squeezed Lemon Juice",
        "quantity": "Squeeze of 1/2 lemon",
        "iron": 0.0, "vit_c": 20.0, "b6": 0.0, "folate": 2.0, "b12": 0.0,
        "veg": True, "vegan": True,
        "cuisines": ["Indian", "Western", "Mediterranean"], "categories": ["booster"]
    },
    {
        "name": "Fresh Orange Wedges",
        "quantity": "1 medium orange",
        "iron": 0.1, "vit_c": 70.0, "b6": 0.1, "folate": 30.0, "b12": 0.0,
        "veg": True, "vegan": True,
        "cuisines": ["Western", "Mediterranean"], "categories": ["booster"]
    },
    {
        "name": "Fresh Sliced Guava",
        "quantity": "1 medium",
        "iron": 0.3, "vit_c": 125.0, "b6": 0.1, "folate": 22.0, "b12": 0.0,
        "veg": True, "vegan": True,
        "cuisines": ["Indian", "Western"], "categories": ["booster"]
    },
    {
        "name": "Red Bell Pepper Slices",
        "quantity": "1/2 cup raw",
        "iron": 0.4, "vit_c": 95.0, "b6": 0.2, "folate": 18.0, "b12": 0.0,
        "veg": True, "vegan": True,
        "cuisines": ["Western", "Mediterranean"], "categories": ["booster"]
    }
]

def generate_dynamic_meal_plan(iron_gap, diet_pref, cuisine_pref, active_genes, rda):
    # 1. Check for iron overload risk
    is_overload_risk = False
    for gene in active_genes:
        if gene["gene"] in ["HFE", "TFR2"]:
            is_overload_risk = True
            break
            
    # G6PD check
    is_g6pd_active = False
    for gene in active_genes:
        if gene["gene"] == "G6PD":
            is_g6pd_active = True
            break

    # Hepcidin and Absorption block check
    needs_high_vit_c = False
    for gene in active_genes:
        if gene["gene"] in ["TMPRSS6", "CYBRD1"]:
            needs_high_vit_c = True
            break

    # Target iron calculations
    if is_overload_risk:
        target_iron_goal = 0.0 # Only maintenance iron
    else:
        target_iron_goal = iron_gap

    # 2. Filter food database based on vegetarian/vegan preference
    pool = DYNAMIC_FOOD_DB.copy()
    if diet_pref == "Vegetarian":
        pool = [f for f in pool if f["veg"]]
    elif diet_pref == "Vegan":
        pool = [f for f in pool if f["vegan"]]

    # If overload risk, remove beef/red meat
    if is_overload_risk:
        pool = [f for f in pool if "beef" not in f["name"].lower() and "steak" not in f["name"].lower()]

    # Helper to select foods prioritized by cuisine
    def select_prioritized(category, cuisine):
        cat_foods = [f for f in pool if category in f["categories"]]
        def sort_key(x):
            c_val = 0 if cuisine in x["cuisines"] else 1
            d_val = 0 if (diet_pref == "Any" and not x["veg"]) else 1
            return (c_val, d_val)
        sorted_foods = sorted(cat_foods, key=sort_key)
        return sorted_foods

    # Select foods
    bfs = select_prioritized("breakfast", cuisine_pref)
    mains = select_prioritized("lunch", cuisine_pref)
    snacks = select_prioritized("snack", cuisine_pref)
    boosters = select_prioritized("booster", cuisine_pref)

    bf = bfs[0] if bfs else None
    
    # Select distinct mains for lunch and dinner
    lm = mains[0] if mains else None
    dm = mains[1] if len(mains) > 1 else (mains[0] if mains else None)
    sn = snacks[0] if snacks else None

    # Vitamin C boosters
    # We want different boosters for breakfast, lunch, and dinner
    bf_b = boosters[0] if boosters else None
    lm_b = boosters[1] if len(boosters) > 1 else (boosters[0] if boosters else None)
    dm_b = boosters[2] if len(boosters) > 2 else (boosters[0] if boosters else None)

    # 3. Portions scaling & Arithmetic
    # Calculate base iron in plan
    base_iron = 0.0
    for f in [bf, lm, dm, sn, bf_b, lm_b, dm_b]:
        if f:
            base_iron += f["iron"]

    multiplier = 1.0
    if not is_overload_risk and target_iron_goal > base_iron:
        # Scale up portions of the lunch and dinner mains
        multiplier = round(target_iron_goal / base_iron, 1)
        multiplier = max(1.0, min(2.0, multiplier)) # Cap at 2.0x

    # Calculate actual intake values
    plan_iron = 0.0
    plan_vit_c = 0.0
    plan_b6 = 0.0
    plan_folate = 0.0
    plan_b12 = 0.0

    for f in [bf, sn, bf_b, lm_b, dm_b]:
        if f:
            plan_iron += f["iron"]
            plan_vit_c += f["vit_c"]
            plan_b6 += f["b6"]
            plan_folate += f["folate"]
            plan_b12 += f["b12"]

    # Add scaled mains
    for f in [lm, dm]:
        if f:
            plan_iron += f["iron"] * multiplier
            plan_vit_c += f["vit_c"] * multiplier
            plan_b6 += f["b6"] * multiplier
            plan_folate += f["folate"] * multiplier
            plan_b12 += f["b12"] * multiplier

    # Enhance Vitamin C booster if hepcidin block active
    extra_vit_c_note = ""
    if needs_high_vit_c:
        # Ensure we double the vit C boosters
        plan_vit_c += (bf_b["vit_c"] if bf_b else 0) + (lm_b["vit_c"] if lm_b else 0)
        extra_vit_c_note = " (Double portions of boosters included for genetic barrier override)"

    # Format output strings
    mult_str = f" ({multiplier}x portion)" if multiplier > 1.0 else ""
    
    plan_title = f"🟢 Normal Maintenance Plan" if is_overload_risk else f"🟡 Balanced Iron Plan ({cuisine_pref})" if target_iron_goal < 6 else f"🟠 High Iron Support ({cuisine_pref})"
    
    breakfast_str = f"{bf['name']} - {bf['quantity']} [{bf['iron']}mg Fe]" if bf else "Oats with water"
    if bf_b:
        breakfast_str += f" + paired with {bf_b['name']} ({bf_b['quantity']}) [{bf_b['vit_c']}mg Vit C]"

    lunch_str = f"{lm['name']}{mult_str} - {lm['quantity']} [{round(lm['iron']*multiplier, 1)}mg Fe]" if lm else "Lentils"
    if lm_b:
        lunch_str += f" + paired with {lm_b['name']} ({lm_b['quantity'] if not needs_high_vit_c else 'Double Portion'}) [{round(lm_b['vit_c'] * (2 if needs_high_vit_c else 1), 1)}mg Vit C]"

    dinner_str = f"{dm['name']}{mult_str} - {dm['quantity']} [{round(dm['iron']*multiplier, 1)}mg Fe]" if dm else "White beans"
    if dm_b:
        dinner_str += f" + paired with {dm_b['name']} ({dm_b['quantity'] if not needs_high_vit_c else 'Double Portion'}) [{round(dm_b['vit_c'] * (2 if needs_high_vit_c else 1), 1)}mg Vit C]"

    snack_str = f"{sn['name']} - {sn['quantity']} [{sn['iron']}mg Fe]" if sn else "Pumpkin seeds"

    # Notes
    note_parts = []
    if is_overload_risk:
        note_parts.append("⚠️ Excess iron warning: BMP6/HFE sensing active. Limit overall iron; avoid supplements.")
    else:
        note_parts.append(f"🎯 Target Iron Gap: {target_iron_goal:.1f}mg | Combined Plan Iron: {plan_iron:.1f}mg ({'Goal Met ✅' if plan_iron >= target_iron_goal else 'Approached limit'})")
    
    note_parts.append(f"🍊 Synergy pairing: Vitamin C total is {plan_vit_c:.1f}mg{extra_vit_c_note}.")
    if is_g6pd_active:
        note_parts.append("🚨 G6PD pathway active: Broad/fava beans completely omitted from all recipes.")
    
    note = " | ".join(note_parts)

    return {
        "title": plan_title,
        "breakfast": breakfast_str,
        "lunch": lunch_str,
        "snack": snack_str,
        "dinner": dinner_str,
        "note": note,
        "arithmetic": {
            "iron_gap_mg": round(target_iron_goal, 1),
            "plan_iron_mg": round(plan_iron, 1),
            "plan_vit_c_mg": round(plan_vit_c, 1),
            "plan_b6_mg": round(plan_b6, 2),
            "plan_folate_mcg": round(plan_folate, 1),
            "plan_b12_mcg": round(plan_b12, 1),
            "portion_scale": multiplier
        }
    }

@app.post("/api/predict")
def api_predict(data: PatientInput):
    if not MODELS["is_trained"]:
        raise HTTPException(status_code=500, detail="Pipeline models not trained yet.")

    try:
        pheno = data.phenotype
        snps = data.snps
        
        # 1. Build features row
        features_row = {}
        for feat in MODELS["feature_names"]:
            if hasattr(pheno, feat):
                features_row[feat] = getattr(pheno, feat)
            elif feat in snps:
                features_row[feat] = float(snps[feat])
            elif feat == 'PRS':
                # Re-calculate PRS for input SNPs
                gwas_sig = MODELS["gwas_sig"]
                gwas_sig_dict = dict(zip(gwas_sig['SNPS'], gwas_sig['Beta']))
                prs_sum = 0.0
                for s_id, dosage in snps.items():
                    if s_id in gwas_sig_dict:
                        prs_sum += float(dosage) * gwas_sig_dict[s_id]
                features_row[feat] = prs_sum
            elif feat == 'PRS_category_enc':
                gwas_sig = MODELS["gwas_sig"]
                gwas_sig_dict = dict(zip(gwas_sig['SNPS'], gwas_sig['Beta']))
                prs_sum = 0.0
                for s_id, dosage in snps.items():
                    if s_id in gwas_sig_dict:
                        prs_sum += float(dosage) * gwas_sig_dict[s_id]
                p25 = MODELS["prs_p25"]
                p75 = MODELS["prs_p75"]
                prs_cat = 'Low' if prs_sum < p25 else 'Moderate' if prs_sum <= p75 else 'High'
                features_row[feat] = 0.0 if prs_cat == 'Low' else 1.0 if prs_cat == 'Moderate' else 2.0
            else:
                # Default to NHANES median or 0
                features_row[feat] = MODELS["nhanes_median"].get(feat, 0.0)

        feature_df = pd.DataFrame([features_row])[MODELS["feature_names"]]

        # 2. Predict using XGBoost
        pred_hb = float(MODELS["regressor"].predict(feature_df)[0])
        pred_prob = float(MODELS["classifier"].predict_proba(feature_df)[0, 1])
        pred_anemia = int(MODELS["classifier"].predict(feature_df)[0])

        # Recalculate PRS cat for reporting
        prs_val = features_row.get('PRS', 0.0)
        p25 = MODELS["prs_p25"]
        p75 = MODELS["prs_p75"]
        prs_category = 'Low' if prs_val < p25 else 'Moderate' if prs_val <= p75 else 'High'

        # 3. Anemia Risk Score
        hb_pen = max(0, (13.0 - pred_hb) / 13.0) * 0.15
        prs_w = {'High': 0.25, 'Moderate': 0.10, 'Low': 0.0}.get(prs_category, 0.0)
        raw_score = pred_prob * 0.60 + prs_w + hb_pen
        risk_score = round(min(100.0, raw_score * 100), 1)

        # Risk classification
        if risk_score >= 65 or (risk_score >= 50 and prs_category == 'High'):
            overall_risk = 'HIGH RISK'
        elif risk_score >= 35 or prs_category == 'High':
            overall_risk = 'MODERATE RISK'
        else:
            overall_risk = 'LOW RISK'

        # 4. Pathway Burden and Prescription Engine
        active_genes_list = []
        gwas_sig = MODELS["gwas_sig"]
        snp_bio = MODELS["snp_bio"]
        for s_id, dosage in snps.items():
            if dosage >= 0.5:
                # Match to GWAS beta and gene
                g_rows = gwas_sig[gwas_sig['SNPS'] == s_id]
                beta = float(g_rows['Beta'].values[0]) if len(g_rows) > 0 else 0.1
                gene_rows = snp_bio[snp_bio['SNPS'] == s_id]
                gene = gene_rows['gene_clean'].values[0] if len(gene_rows) > 0 else "Unknown"
                active_genes_list.append({
                    "gene": gene,
                    "snp": s_id,
                    "beta": beta,
                    "dosage": float(dosage)
                })

        # Sort by impact
        active_genes_list.sort(key=lambda x: abs(x['beta']) * x['dosage'], reverse=True)

        burden_scores, ranked_axes = compute_pathway_burden_scores(active_genes_list)
        priority_nutrients, priority_avoidances, conflicts, clinical_flags = \
            build_nutrient_requirement_profile(active_genes_list, burden_scores, ranked_axes)

        hb_cat = 'SEVERE' if pred_hb < 10 else 'HIGH' if pred_hb < 12 else 'MODERATE' if pred_hb < 13 else 'NORMAL'
        rda = 18 if pheno.RIAGENDR == 1 else 8
        diet_gap = round(max(0.0, rda - pheno.DR1TIRON), 1)

        # Generate dynamic, composable meal plan
        plan = generate_dynamic_meal_plan(
            iron_gap=diet_gap,
            diet_pref=data.diet_pref,
            cuisine_pref=data.cuisine_pref,
            active_genes=active_genes_list,
            rda=rda
        )
        override_reason = f"Dynamic plan generated for {data.cuisine_pref} ({data.diet_pref}) matching a {diet_gap}mg iron gap."

        # Custom food lists
        base_iron_foods = (FOOD_DB['HIGH_IRON'] if hb_cat in ('SEVERE','HIGH') else
                           FOOD_DB['MODERATE_IRON'] if hb_cat == 'MODERATE' else
                           FOOD_DB['MAINTENANCE'])
        base_recs = [f"Eat MORE {f[0]}: {f[1]} per day — {f[3]}" for f in base_iron_foods[:4]]
        
        # Gene specific recommendations
        gene_specific_foods = {}
        for gene_entry in active_genes_list[:8]:
            gene = gene_entry['gene']
            kb = get_kb_entry(gene)
            if not kb:
                continue
            foods = []
            for rec in kb.get('increase', [])[:2]:
                foods.append(f"[{rec['evidence']}] {rec['nutrient']} — {rec['rationale']}")
            if foods:
                gene_specific_foods[gene] = foods

        # Avoid list
        avoid_items = [f"{f[0]}: {f[1]}" for f in FOOD_DB['AVOID_STANDARD']]
        for a in priority_avoidances[:6]:
            item = f"[{a['evidence']}] {a['nutrient']} (genes: {', '.join(a['genes'][:2])})"
            if item not in avoid_items:
                avoid_items.append(item)

        # Generate metabolic explanations
        pathway_explanations = []
        for gene_entry in active_genes_list[:6]:
            gene = gene_entry['gene']
            kb_key = None
            if gene in PATHWAY_KB:
                kb_key = gene
            else:
                for k in PATHWAY_KB:
                    if gene.startswith(k) or k.startswith(gene):
                        kb_key = k
                        break
            if kb_key:
                kb = PATHWAY_KB[kb_key]
                pathway_explanations.append({
                    "gene": gene,
                    "snp": gene_entry["snp"],
                    "dosage": gene_entry["dosage"],
                    "beta": gene_entry["beta"],
                    "pathway": kb["pathway"],
                    "mechanism": kb["mechanism"],
                    "anemia_link": kb["anemia_link"],
                    "diet_impact": kb["diet_impact"]
                })
            else:
                # fallback
                pathway_explanations.append({
                    "gene": gene,
                    "snp": gene_entry["snp"],
                    "dosage": gene_entry["dosage"],
                    "beta": gene_entry["beta"],
                    "pathway": "Erythropoiesis / Iron Metabolism Pathway",
                    "mechanism": f"{gene} is associated with Hb levels with beta={gene_entry['beta']:.4f}.",
                    "anemia_link": "GWAS-associated variant.",
                    "diet_impact": "Maintain optimal iron, folate, B12, and Vitamin C intake as a general health precaution."
                })

        priority_label = ('Urgent — see doctor immediately' if hb_cat == 'SEVERE' else
                          'Daily iron-rich meals + Vit C pairing, every meal' if hb_cat == 'HIGH' else
                          'Consistent iron diet + 3-monthly monitoring' if hb_cat == 'MODERATE' else
                          'Maintain healthy habits + annual check-up')

        return {
            "predicted_hb": round(pred_hb, 2),
            "anemia_predicted": pred_anemia,
            "anemia_probability": round(pred_prob, 3),
            "prs_value": round(prs_val, 4),
            "prs_category": prs_category,
            "anemia_risk_score": risk_score,
            "overall_risk": overall_risk,
            "priority": priority_label,
            "rda_mg": rda,
            "daily_iron_gap_mg": diet_gap,
            "meal_plan": {
                "title": plan["title"],
                "breakfast": plan["breakfast"],
                "lunch": plan["lunch"],
                "snack": plan.get("snack", "Yogurt + fruit"),
                "dinner": plan["dinner"],
                "note": plan["note"],
                "arithmetic": plan.get("arithmetic", {})
            },
            "meal_override_reason": override_reason,
            "base_foods_eat_more": base_recs,
            "gene_specific_foods": gene_specific_foods,
            "vitamin_c_boosters": [f"{f[0]}: {f[1]} — {f[2]}" for f in FOOD_DB['VITAMIN_C'][:3]],
            "foods_to_avoid": avoid_items,
            "priority_nutrients": [n['nutrient'] for n in priority_nutrients[:8]],
            "priority_avoidances": [a['nutrient'] for a in priority_avoidances[:6]],
            "conflicts": [c['resolution'] for c in conflicts],
            "clinical_flags": [f"{f['gene']}: {f['note']}" for f in clinical_flags],
            "pathway_burden_scores": burden_scores,
            "pathway_explanations": pathway_explanations,
            "active_genes": active_genes_list
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

class NutritionInput(BaseModel):
    diet_pref: str  # Vegan, Vegetarian, Non-Vegetarian
    allergies: List[str]  # e.g. ["Nuts"]
    intake_pattern: str  # High Carb Diet, High Protein Diet, High Fat Diet, Balanced Diet, Low Carb Diet
    most_consumed: str  # Rice/Bread/Pasta, Chicken/Eggs/Paneer, Fried foods/Cheese/Butter

NUTRITION_FOODS = [
    # Grains/Carbs
    {"name": "Brown Rice", "type": "carb", "veg": True, "vegan": True, "allergens": [], "desc": "Fiber-rich complex carb for sustained energy"},
    {"name": "Quinoa", "type": "protein_carb", "veg": True, "vegan": True, "allergens": [], "desc": "Complete plant protein + complex carb"},
    {"name": "Oats", "type": "carb", "veg": True, "vegan": True, "allergens": ["Gluten"], "desc": "Heart-healthy soluble fiber source"},
    {"name": "Sweet Potato", "type": "carb", "veg": True, "vegan": True, "allergens": [], "desc": "High fiber and Vitamin A complex carb"},
    {"name": "Whole Wheat Sourdough", "type": "carb", "veg": True, "vegan": True, "allergens": ["Gluten"], "desc": "Complex carbs"},
    # Proteins
    {"name": "Organic Tofu", "type": "protein", "veg": True, "vegan": True, "allergens": ["Soy"], "desc": "High plant-based protein"},
    {"name": "Tempeh", "type": "protein", "veg": True, "vegan": True, "allergens": ["Soy"], "desc": "Fermented high-protein soy product"},
    {"name": "Chickpeas & Lentils", "type": "protein_carb", "veg": True, "vegan": True, "allergens": [], "desc": "Excellent protein and fiber combination"},
    {"name": "Paneer", "type": "protein", "veg": True, "vegan": False, "allergens": ["Dairy"], "desc": "High dairy protein and calcium"},
    {"name": "Greek Yogurt", "type": "protein", "veg": True, "vegan": False, "allergens": ["Dairy"], "desc": "Probiotic-rich high protein dairy"},
    {"name": "Organic Eggs", "type": "protein", "veg": True, "vegan": False, "allergens": ["Eggs"], "desc": "Complete animal protein source"},
    {"name": "Grilled Chicken Breast", "type": "protein", "veg": False, "vegan": False, "allergens": [], "desc": "Lean high-quality animal protein"},
    {"name": "Wild-Caught Grilled Salmon", "type": "protein_fat", "veg": False, "vegan": False, "allergens": ["Seafood"], "desc": "Rich in Omega-3 fatty acids and protein"},
    {"name": "White Fish (Cod)", "type": "protein", "veg": False, "vegan": False, "allergens": ["Seafood"], "desc": "Ultra-lean marine protein"},
    # Fats
    {"name": "Avocado", "type": "fat", "veg": True, "vegan": True, "allergens": [], "desc": "Heart-healthy monounsaturated fats"},
    {"name": "Almonds & Walnuts", "type": "fat_protein", "veg": True, "vegan": True, "allergens": ["Nuts"], "desc": "Brain-healthy fats and protein"},
    {"name": "Chia & Flax Seeds", "type": "fat", "veg": True, "vegan": True, "allergens": [], "desc": "Omega-3 rich plant seeds"},
    {"name": "Extra Virgin Olive Oil", "type": "fat", "veg": True, "vegan": True, "allergens": [], "desc": "Premium monounsaturated cooking fat"},
    {"name": "Cheese / Butter", "type": "fat", "veg": True, "vegan": False, "allergens": ["Dairy"], "desc": "Saturated fats (consume in moderation)"}
]

@app.post("/api/nutrition-recommend")
def api_nutrition_recommend(data: NutritionInput):
    try:
        # 1. Filter Foods by Diet Preference
        pool = NUTRITION_FOODS.copy()
        if data.diet_pref == "Vegan":
            pool = [f for f in pool if f["vegan"]]
        elif data.diet_pref == "Vegetarian":
            pool = [f for f in pool if f["veg"]]

        # 2. Filter Foods by Allergies
        allergy_cleaned = []
        for f in pool:
            is_allergic = False
            for allergy in data.allergies:
                if allergy.lower() in [a.lower() for a in f["allergens"]]:
                    is_allergic = True
                    break
            if not is_allergic:
                allergy_cleaned.append(f)

        # 3. Identify Foods to Avoid
        avoid_recs = []
        if data.diet_pref == "Vegan":
            avoid_recs.append("All animal products (meat, fish, poultry, eggs, dairy, honey)")
        elif data.diet_pref == "Vegetarian":
            avoid_recs.append("All meat, fish, seafood, and poultry")
        for a in data.allergies:
            avoid_recs.append(f"{a} (Allergen - avoid completely)")

        # Pattern-specific rules
        pattern_notes = []
        include_foods = allergy_cleaned.copy()
        
        # High Carb Diet or Carb consumption most -> Limit refined carbs, boost protein & fiber
        if data.intake_pattern == "High Carb Diet" or "Rice/Bread/Pasta" in data.most_consumed:
            avoid_recs.append("Refined grains, white bread, white pasta, sugary cereals")
            pattern_notes.append("⚠️ High carbohydrate intake pattern: Focus on high-protein and high-fiber foods to support insulin sensitivity and prevent blood sugar spikes.")
            include_foods = sorted(include_foods, key=lambda x: 0 if x["type"] in ["protein", "protein_carb", "fat_protein"] else 1)
        
        # High Fat Diet or Fat consumption most -> Avoid trans/saturated fats, encourage healthy unsaturated fats
        if data.intake_pattern == "High Fat Diet" or "Fried foods/Cheese/Butter" in data.most_consumed:
            avoid_recs.append("Fried foods, trans fats, fatty cuts of meat, excessive butter/cheese")
            pattern_notes.append("⚠️ High saturated fat intake pattern: Swapped heavy fats for heart-healthy unsaturated fats (Avocado, Chia Seeds, Extra Virgin Olive Oil). Focus on lighter steamed/grilled meals.")
            include_foods = [f for f in include_foods if "cheese" not in f["name"].lower() and "butter" not in f["name"].lower()]
            include_foods = sorted(include_foods, key=lambda x: 0 if x["type"] == "fat" and x["name"] in ["Avocado", "Extra Virgin Olive Oil"] else 1)

        # Low Carb Diet -> Emphasize proteins and healthy fats, minimize grains
        if data.intake_pattern == "Low Carb Diet":
            avoid_recs.append("High glycemic index grains, potatoes, starchy tubers, white rice")
            pattern_notes.append("⚡ Low carbohydrate pattern: Selected protein-dense and fat-dense meals while keeping starches minimal.")
            include_foods = [f for f in include_foods if f["type"] not in ["carb"]]

        # Low Protein / High Protein Habit
        if data.intake_pattern == "High Protein Diet" or "Chicken/Eggs/Paneer" in data.most_consumed:
            pattern_notes.append("💪 High Protein Target: Emphasized amino-acid dense protein options to support muscle recovery and cellular repair.")
            include_foods = sorted(include_foods, key=lambda x: 0 if x["type"] in ["protein", "protein_carb", "fat_protein", "protein_fat"] else 1)

        # 4. Suggested Meal Plan Composition
        def get_food_by_type(types, exclude_names=[]):
            for f in include_foods:
                if f["type"] in types and f["name"] not in exclude_names:
                    return f
            for f in include_foods:
                if f["name"] not in exclude_names:
                    return f
            return {"name": "Fresh Steamed Greens", "desc": "Light steamed vegetables"}

        bf_food = get_food_by_type(["carb", "protein_carb", "protein"])
        lunch_food = get_food_by_type(["protein", "protein_carb", "protein_fat"], [bf_food["name"]])
        dinner_food = get_food_by_type(["protein", "protein_fat", "fat"], [bf_food["name"], lunch_food["name"]])
        snack_food = get_food_by_type(["fat_protein", "fat"], [bf_food["name"], lunch_food["name"], dinner_food["name"]])

        breakfast = f"Baked {bf_food['name']} bowl or porridge - paired with water or unsweetened tea/coffee. ({bf_food['desc']})"
        lunch = f"Hearty {lunch_food['name']} salad/wrap - with mixed baby greens and olive oil dressing. ({lunch_food['desc']})"
        dinner = f"Grilled/Steamed {dinner_food['name']} - served alongside stir-fried broccoli and asparagus. ({dinner_food['desc']})"
        snack = f"{snack_food['name']} ({snack_food['desc']})"

        drink_options = ["Purified Lemon Water", "Warm Green Tea", "Hibiscus Tea"]
        if "Dairy" not in data.allergies and data.diet_pref != "Vegan":
            drink_options.append("Kefir or Buttermilk")
        elif "Soy" not in data.allergies:
            drink_options.append("Unsweetened Soy Milk")
        elif "Nuts" not in data.allergies:
            drink_options.append("Unsweetened Almond Milk")
        drinks = " or ".join(drink_options[:3])

        if data.intake_pattern == "Low Carb Diet":
            cal_est = "1500 - 1700 kcal / day (Weight-loss & insulin optimization)"
        elif data.intake_pattern == "High Protein Diet":
            cal_est = "2000 - 2200 kcal / day (Anabolic & muscle repair support)"
        elif data.intake_pattern == "High Fat Diet":
            cal_est = "1700 - 1900 kcal / day (Caloric maintenance & fat modification)"
        else:
            cal_est = "1800 - 2000 kcal / day (Standard daily balanced metabolic requirement)"

        adv_list = []
        if pattern_notes:
            adv_list.extend(pattern_notes)
        else:
            adv_list.append("Maintain a consistent intake of whole foods, chew slowly, and eat within a 10-hour daily window.")
        adv_list.append("Ensure optimal hydration: Drink 2.5 to 3 liters of water daily, separated from meals by 30 minutes.")

        return {
            "dietary_type": f"{data.diet_pref} - {data.intake_pattern}",
            "foods_to_include": [f"{f['name']}: {f['desc']}" for f in include_foods[:6]],
            "foods_to_avoid": avoid_recs,
            "suggested_meal_plan": {
                "breakfast": breakfast,
                "lunch": lunch,
                "dinner": dinner,
                "snacks": snack,
                "drinks": drinks
            },
            "nutritional_benefits": [
                "Boosts bioavailability of clean protein and essential fatty acids",
                "Enhances digestive rest through dietary fiber integration",
                "Substitutes inflammatory trigger inputs with clean, single-ingredient whole foods"
            ],
            "calorie_estimate": cal_est,
            "nutritional_advice": adv_list
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Nutrition compilation failed: {str(e)}")
app.mount("/", StaticFiles(directory="/Users/yashna/idpfood/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

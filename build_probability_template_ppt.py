from pathlib import Path
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent
TEMPLATE = Path("/home/mrudula/Downloads/22AIE301_Template_Final_Review.pptx")
OUT = ROOT / "Materials_Adversarial_Final_Project_Review.pptx"
DAG = ROOT / "artifacts" / "probability_dag.png"
CHART = ROOT / "results" / "final" / "figures" / "attack_strategy_comparison" / "budget_trajectories.png"

NAVY = RGBColor(0x1D, 0x2D, 0x50)
CYAN = RGBColor(0x06, 0xB6, 0xD4)
ORANGE = RGBColor(0xF5, 0x9E, 0x0B)
GREEN = RGBColor(0x16, 0xA3, 0x4A)
RED = RGBColor(0xDC, 0x26, 0x26)
GRAY = RGBColor(0x47, 0x55, 0x69)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def set_header(shape, section):
    tf = shape.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = "22AIE301 PROBABILISTIC REASONING — FINAL PROJECT REVIEW"
    r.font.size = Pt(9); r.font.bold = True; r.font.color.rgb = CYAN
    p = tf.add_paragraph()
    r = p.add_run(); r.text = section
    r.font.size = Pt(24); r.font.bold = True; r.font.color.rgb = NAVY


def set_footer(shape, slide_no):
    shape.text = f"22AIE301 Course Project Final Review  |  Slide {slide_no} of 10"
    p = shape.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    for r in p.runs:
        r.font.size = Pt(8); r.font.color.rgb = GRAY


def set_card(shape, heading, bullets, font_size=13.5, accent=CYAN):
    tf = shape.text_frame
    tf.clear(); tf.word_wrap = True
    tf.margin_left = Inches(0.18); tf.margin_right = Inches(0.18)
    tf.margin_top = Inches(0.15); tf.margin_bottom = Inches(0.12)
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = heading
    r.font.size = Pt(17); r.font.bold = True; r.font.color.rgb = accent
    for text in bullets:
        p = tf.add_paragraph(); p.level = 0; p.text = text
        p.font.size = Pt(font_size); p.font.color.rgb = NAVY
        p.space_before = Pt(7); p.line_spacing = 1.05


def add_caption(slide, x, y, w, h, text, size=10, color=GRAY, bold=False):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame; tf.clear(); tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.color.rgb = color; r.font.bold = bold
    return box


prs = Presentation(TEMPLATE)

# Slide 1
s = prs.slides[0]
tf = s.shapes[1].text_frame; tf.clear(); tf.word_wrap = True
lines = [
    ("22AIE301 PROBABILISTIC REASONING — FINAL PROJECT REVIEW", 10, CYAN, True),
    ("Adversarial Robustness in Scientific Sequence Models", 27, WHITE, True),
    ("Track Declaration:  [  ] Track I   |   [✓] Track II: Journal / Conference Paper", 12, ORANGE, True),
    ("Materials and bio-cyber project review with probability-aware attack evaluation", 15, WHITE, False),
    ("Group No: [Add group number]", 12, WHITE, False),
    ("Team Members: [Add names and roll numbers]", 12, WHITE, False),
    ("Mentor: [Add name and affiliation]", 12, WHITE, False),
]
for i, (text, size, color, bold) in enumerate(lines):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    r = p.add_run(); r.text = text; r.font.size = Pt(size); r.font.color.rgb = color; r.font.bold = bold
    p.space_after = Pt(8 if i in (0,1,2) else 4)

# Slide 2
s = prs.slides[1]; set_header(s.shapes[0], "1. Problem Statement, Research Scope & Objectives"); set_footer(s.shapes[2], 2)
set_card(s.shapes[3], "Project Problem & Motivation", [
    "Scientific models often consume sequence representations such as polymer SMILES and biological relationship encodings.",
    "A model may predict accurately on clean data but change sharply after small token-level perturbations.",
    "The project builds attacks, evaluates defenses, and audits whether observed vulnerability reflects the model, the representation, or an experimental confound.",
    "The materials case focuses on polymer property prediction; the bio-cyber case provides a second domain for attack comparison."
], 13)
set_card(s.shapes[4], "Core Project Objectives", [
    "1. Train and audit Transformer-based scientific sequence predictors.",
    "2. Implement Random, attribution-guided, MCMC, evolutionary, and generative perturbation searches.",
    "3. Compare attack families fairly using matched query budgets, validity filters, and paired source examples.",
    "4. Add probability estimates so observed success rates include uncertainty instead of relying only on maxima."
], 13)

# Slide 3
s = prs.slides[2]; set_header(s.shapes[0], "2. Project Pipeline and Probabilistic Model"); set_footer(s.shapes[2], 3)
set_card(s.shapes[3], "End-to-End Project Flow", ["Input: scientific sequence plus target property.", "Attack engine proposes altered sequences under a fixed budget.", "Validity checks filter candidates before the frozen predictor measures drift."], 10.8)
s.shapes.add_picture(str(DAG), s.shapes[3].left+Inches(0.2), s.shapes[3].top+Inches(2.05), width=s.shapes[3].width-Inches(0.4))
set_card(s.shapes[4], "Probability Overlay", [
    "P(Q,V,L,D,S | A,B,x) =",
    "P(Q | A,x) P(V | Q,x) P(L | V,x)",
    "× P(D | L,Q,B,x) P(S | V,L,D).",
    "A counted success must be valid, label-preserving, and exceed drift threshold τ."
], 12)
set_card(s.shapes[5], "Datasets and Models", [
    "Tg forensic audit: 247 cleaned polymers; split 204 train, 37 validation, 6 test.",
    "Bandgap model: 2,946 train, 631 validation, 632 sealed test examples.",
    "Transformer: token + positional embeddings, two encoder layers, four heads, mean pooling.",
    "Graph and simple length controls test whether sequence shortcuts drive the result."
], 12)

# Slide 4
s = prs.slides[3]; set_header(s.shapes[0], "3. Attack Methods and Probability-Aware Inference"); set_footer(s.shapes[2], 4)
set_card(s.shapes[3], "Attack Families", [
    "Random: establishes a simple search baseline.",
    "Attribution-guided: edits high-, low-, or randomly ranked tokens.",
    "MCMC: accepts or rejects stochastic proposals while exploring high-drift regions.",
    "Evolutionary: mutates and selects a population of candidates.",
    "Generative: samples from a trained or untrained generator, then passes candidates through validity checks."
], 12.5)
set_card(s.shapes[4], "Probability Emphasis", [
    "For k successful sources out of n, use p ~ Beta(1,1) and p | k,n ~ Beta(k+1,n−k+1).",
    "At bio-cyber budget 50, Evolutionary records 17/30 successes: posterior mean 0.562, 95% interval [0.391, 0.727].",
    "MCMC records 14/30: 0.469 [0.302, 0.640]; Random records 12/30: 0.406 [0.245, 0.578].",
    "The overlap shows why the probability analysis supports the project without replacing its engineering and forensic evidence."
], 12.5, ORANGE)

# Slide 5
s = prs.slides[4]; set_header(s.shapes[0], "4. Model Architecture, Tools and Defense Strategy"); set_footer(s.shapes[2], 5)
set_card(s.shapes[3], "Transformer Prediction System", [
    "Regex tokenization converts PSMILES into token IDs; sequences are padded to a controlled maximum length.",
    "Token and learned positional embeddings feed a two-layer, four-head Transformer encoder.",
    "Mean pooling produces a fixed representation; an MLP outputs Tg or bandgap prediction.",
    "PyTorch handles modeling, RDKit checks chemistry syntax, and SciPy supports statistical comparison."
], 12.5)
set_card(s.shapes[4], "Defense and Audit Strategy", [
    "Adversarial training augments clean examples with generated perturbations.",
    "Representation-consistency training penalizes prediction changes for valid equivalent sequences.",
    "Frozen checkpoints, attack banks, scalers, and validators prevent post-hoc test tuning.",
    "Phase 6 defense selection remains incomplete, so the deck does not claim a final defense improvement."
], 12.5)

# Slide 6
s = prs.slides[5]; set_header(s.shapes[0], "5. Core Artifact Demonstration & Performance Visuals"); set_footer(s.shapes[2], 6)
set_card(s.shapes[3], "How Attack Strength Changes with Query Budget", [], 12)
s.shapes.add_picture(str(CHART), s.shapes[3].left+Inches(0.25), s.shapes[3].top+Inches(0.7), width=s.shapes[3].width-Inches(0.5), height=s.shapes[3].height-Inches(1.1))
set_card(s.shapes[4], "Key Metrics & Insights", [
    "Benchmark cells: n = 30 sources.",
    "Matched budgets: 5, 20, and 50 target queries.",
    "Evolutionary and MCMC improve substantially as the search budget grows.",
    "In materials, the trained generator produced 0 valid proposals from 22,500 attempts.",
    "That zero-validity result diagnoses generator failure; it does not prove that generative attacks are weak in principle.",
    "Probability intervals keep the 30-source comparisons appropriately cautious."
], 11.5, ORANGE)

# Slide 7
s = prs.slides[6]; set_header(s.shapes[0], "6. Main Findings, Controls and Scientific Boundaries"); set_footer(s.shapes[2], 7)
set_card(s.shapes[3], "What the Project Found", [
    "The Transformer shows real sensitivity to token-level and representation-level changes.",
    "A controlled forensic audit refuted the earlier claim that length change alone causes the observed drift.",
    "A length-only baseline beat the Tg Transformer on validation: 76.77 K versus 79.04 K MAE, revealing split and shortcut concerns.",
    "Independent representation-equivalence testing confirmed strong drift across 2,907 valid variants."
], 12)
set_card(s.shapes[4], "What the Evidence Does Not Prove", [
    "RDKit-valid syntax does not guarantee an unchanged polymer or unchanged property label.",
    "Insertion, deletion, and substitution can alter chemistry; they are not pure invariance tests.",
    "Paired Wilcoxon tests establish drift-distribution differences, not chemical attack success probabilities.",
    "Physical-oracle validation, cross-seed stability, and sealed final defense evaluation remain pending."
], 12)

# Slide 8
s = prs.slides[7]; set_header(s.shapes[0], "7. Final Track Deliverable & Assessment Proof"); set_footer(s.shapes[2], 8)
set_card(s.shapes[3], "TRACK SELECTION DECLARATION", ["[  ] TRACK I: INDUSTRY PRODUCT & VERIFICATION     [✓] TRACK II: JOURNAL / CONFERENCE PAPER"], 13, ORANGE)
set_card(s.shapes[4], "REPOSITORY & EXECUTION PROOF", [
    "Executable pipelines cover model training, perturbation generation, matched-budget comparison, and forensic analysis.",
    "Final artifacts include comparison CSVs, statistical JSON, model manifests, attack banks, and rendered figures.",
    "Focused forensic tests recorded 200 passed and 1 skipped; representation-equivalence validation has independent evidence."
], 12)
set_card(s.shapes[5], "LATEX MANUSCRIPT & REVIEW PROOF", [
    "Materials_Adversarial_Detailed_Report.tex and compiled PDF are present.",
    "This presentation combines the full project explanation with a probability-centered LaTeX/TikZ model.",
    "Claims remain qualified because label preservation, physical validation, multi-seed stability, and final defense evidence are incomplete."
], 12, GREEN)

# Slide 9
s = prs.slides[8]; set_header(s.shapes[0], "8. Conclusion, Repository Deliverables & Future Scope"); set_footer(s.shapes[2], 9)
set_card(s.shapes[3], "Key Technical Contributions & Summary", [
    "Built a cross-domain framework for attacking and auditing scientific sequence models.",
    "Implemented multiple search strategies with explicit validity, query, runtime, and provenance accounting.",
    "Used forensic controls to separate real model instability from length, split, and representation confounds.",
    "Added Bayesian success-rate estimates so uncertainty accompanies the engineering results."
], 12.5)
set_card(s.shapes[4], "Repository Status & Future Scope", [
    "Next: add a chemistry-aware label-preservation oracle and independent physical or computational validation.",
    "Repeat matched-budget comparisons across seeds and fit a hierarchical model across sources and domains.",
    "Complete Phase 6 validation-only selection before opening the sealed test set.",
    "Publish conclusions by evidence tier: model behavior, representation invariance, chemical validity, and physical truth."
], 12.5)

# Slide 10
s = prs.slides[9]; set_header(s.shapes[0], "9. Digitally Verifiable Literature & Software References"); set_footer(s.shapes[2], 10)
set_card(s.shapes[3], "Academic, Software & Repository References", [
    "1. Koller, D. & Friedman, N. Probabilistic Graphical Models. MIT Press, 2009.",
    "2. Gelman, A. et al. Bayesian Data Analysis, 3rd ed. CRC Press, 2013.",
    "3. Wilcoxon, F. Individual Comparisons by Ranking Methods. Biometrics Bulletin, 1945.",
    "4. PyTorch documentation: https://pytorch.org/docs/stable/",
    "5. SciPy statistical functions: https://docs.scipy.org/doc/scipy/reference/stats.html",
    "6. Repository evidence: attack_strategy_comparison.csv; statistical_results.json; forensic_diagnostics.json; representation-equivalence results.",
    "7. Materials_Adversarial_Detailed_Report.tex, ARCHITECTURE_FORENSIC.md, and FINAL_ATTACK_STRATEGY_COMPARISON.md."
], 13)

prs.save(OUT)
print(f"Wrote {OUT}")

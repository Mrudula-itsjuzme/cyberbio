from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import uno
from com.sun.star.awt import Size, Point
from com.sun.star.beans import PropertyValue


FILL_SOLID = uno.Enum("com.sun.star.drawing.FillStyle", "SOLID")
LINE_SOLID = uno.Enum("com.sun.star.drawing.LineStyle", "SOLID")
PARA_BREAK = uno.getConstantByName("com.sun.star.text.ControlCharacter.PARAGRAPH_BREAK")


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "materials-adversarial"
OUT = ROOT / "Materials_Adversarial_Formatted_Review.pptx"
TMP_ODP = ROOT / "Materials_Adversarial_Formatted_Review.odp"
CURRENT_DOC = None

W, H = 33300, 18700
NAVY = 0x06111F
INK = 0x0B1D2E
PANEL = 0x10263A
PANEL_2 = 0x132F46
BLUE = 0x2563EB
CYAN = 0x38BDF8
AMBER = 0xF59E0B
GREEN = 0x22C55E
RED = 0xEF4444
MUTED = 0x9CB4C8
WHITE = 0xF8FAFC
LINE = 0x31506A


def prop(name, value):
    p = PropertyValue()
    p.Name = name
    p.Value = value
    return p


def load_json(rel):
    with open(PROJECT / rel, "r", encoding="utf-8") as f:
        return json.load(f)


diag = load_json("results/forensic_diagnostics.json")
phase2b = load_json("results/phase2b_audit.json")
clean = load_json("results/phase2b_clean_eval.json")
trainset = load_json("results/phase2b_trainset_report.json")
phase2c = load_json("results/phase2c_audit.json")
bandgap = load_json("results/models/transformer_regressor/metrics.json")


def connect():
    env = os.environ.copy()
    env["HOME"] = "/tmp"
    proc = subprocess.Popen(
        [
            "libreoffice",
            "--headless",
            "--invisible",
            "--nodefault",
            "--norestore",
            "--accept=pipe,name=codex_ppt_pipe;urp;StarOffice.ComponentContext",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    local_ctx = uno.getComponentContext()
    resolver = local_ctx.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local_ctx
    )
    for _ in range(80):
        try:
            ctx = resolver.resolve(
                "uno:pipe,name=codex_ppt_pipe;urp;StarOffice.ComponentContext"
            )
            smgr = ctx.ServiceManager
            desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
            return proc, desktop
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("Could not connect to LibreOffice")


def set_text(shape, text, size=22, color=WHITE, bold=False, align=0):
    shape.String = text
    shape.TextWordWrap = True
    shape.TextVerticalAdjust = 0
    shape.ParaAdjust = align
    cur = shape.createTextCursor()
    cur.gotoStart(False)
    cur.gotoEnd(True)
    cur.CharHeight = float(size)
    cur.CharColor = color
    cur.CharFontName = "Aptos"
    cur.CharWeight = 150.0 if bold else 100.0


def add_shape(slide, kind, x, y, w, h, fill=INK, line=LINE, radius=0):
    sh = CURRENT_DOC.createInstance(kind)
    sh.Position = Point(x, y)
    sh.Size = Size(w, h)
    if hasattr(sh, "FillStyle"):
        sh.FillStyle = FILL_SOLID
        sh.FillColor = fill
    if hasattr(sh, "LineStyle"):
        sh.LineStyle = LINE_SOLID
        sh.LineColor = line
    if hasattr(sh, "CornerRadius"):
        sh.CornerRadius = radius
    slide.add(sh)
    return sh


def box(slide, x, y, w, h, text="", fill=INK, line=LINE, size=20, color=WHITE, bold=False, align=0):
    sh = add_shape(slide, "com.sun.star.drawing.RectangleShape", x, y, w, h, fill, line)
    set_text(sh, text, size, color, bold, align)
    sh.TextLeftDistance = 450
    sh.TextRightDistance = 450
    sh.TextUpperDistance = 300
    sh.TextLowerDistance = 250
    return sh


def label(slide, x, y, w, text, color=MUTED, size=12, align=0):
    return box(slide, x, y, w, 430, text.upper(), NAVY, NAVY, size, color, True, align)


def title(slide, text, subtitle=None):
    box(slide, 1050, 600, 30800, 900, text, NAVY, NAVY, 27, WHITE, True)
    if subtitle:
        box(slide, 1080, 1540, 22800, 600, subtitle, NAVY, NAVY, 14, MUTED)
    add_shape(slide, "com.sun.star.drawing.RectangleShape", 1050, 2320, 4300, 60, AMBER, AMBER)
    add_shape(slide, "com.sun.star.drawing.RectangleShape", 5450, 2320, 26400, 18, LINE, LINE)


def bullet_text(slide, x, y, w, h, items, size=19):
    sh = box(slide, x, y, w, h, "", NAVY, NAVY, size)
    text = sh.Text
    cur = text.createTextCursor()
    for i, item in enumerate(items):
        if i:
            text.insertControlCharacter(cur, PARA_BREAK, False)
        text.insertString(cur, "• " + item, False)
    cur.gotoStart(False); cur.gotoEnd(True)
    cur.CharHeight = float(size); cur.CharColor = WHITE; cur.CharFontName = "Aptos"
    return sh


def line(slide, x1, y1, x2, y2, color=MUTED, width=70):
    sh = add_shape(slide, "com.sun.star.drawing.LineShape", x1, y1, x2-x1, y2-y1, NAVY, color)
    sh.LineWidth = width
    sh.LineColor = color
    return sh


def chip(slide, x, y, text, color):
    return box(slide, x, y, 3600, 650, text, color, color, 15, 0x061019, True, 1)


def add_slide(doc, name, subtitle=None):
    pages = doc.DrawPages
    slide = pages.insertNewByIndex(pages.Count)
    add_shape(slide, "com.sun.star.drawing.RectangleShape", 0, 0, W, H, NAVY, NAVY)
    add_shape(slide, "com.sun.star.drawing.RectangleShape", 0, 0, 420, H, 0x0C2A3F, 0x0C2A3F)
    title(slide, name, subtitle)
    box(slide, 29200, 17440, 2500, 420, f"{pages.Count:02d}", NAVY, NAVY, 11, MUTED, True, 2)
    box(slide, 1050, 17440, 11800, 420, "Materials adversarial learning forensic review", NAVY, NAVY, 10, 0x6F879A)
    return slide


def section_slide(doc, kicker, headline, subline):
    s = add_slide(doc, kicker)
    box(s, 1800, 5300, 28500, 3900, headline, NAVY, NAVY, 34, WHITE, True, 1)
    box(s, 4500, 9800, 23000, 1200, subline, NAVY, NAVY, 19, MUTED, False, 1)
    add_shape(s, "com.sun.star.drawing.RectangleShape", 8200, 11750, 16800, 80, AMBER, AMBER)
    return s


def metric_card(slide, x, y, w, h, value, label_text, accent=CYAN, note=None):
    add_shape(slide, "com.sun.star.drawing.RectangleShape", x, y, w, h, PANEL, LINE)
    add_shape(slide, "com.sun.star.drawing.RectangleShape", x, y, 180, h, accent, accent)
    box(slide, x + 520, y + 360, w - 900, 760, value, PANEL, PANEL, 27, WHITE, True)
    box(slide, x + 540, y + 1260, w - 1000, 640, label_text, PANEL, PANEL, 14, MUTED)
    if note:
        box(slide, x + 540, y + 1960, w - 1000, 520, note, PANEL, PANEL, 12, accent)


def data_flow(slide):
    lane_y = [3400, 6900, 10400, 13900]
    lane_names = ["DATA", "ATTACKS", "MODEL", "AUDIT"]
    for y, name in zip(lane_y, lane_names):
        label(slide, 1050, y - 540, 3600, name, AMBER if name == "AUDIT" else CYAN)
        add_shape(slide, "com.sun.star.drawing.RectangleShape", 1050, y - 120, 30300, 55, LINE, LINE)

    data = [
        ("Raw table", "741 rows\n32 properties"),
        ("Tg filter", "443 non-null\nTg records"),
        ("Canonicalize", "RDKit\n+ drop conflicts"),
        ("Final split", "247 samples\n204 / 37 / 6"),
    ]
    xs = [1650, 9000, 16350, 23700]
    for i, (a, b) in enumerate(data):
        box(slide, xs[i], lane_y[0], 6100, 1600, a + "\n" + b, PANEL, CYAN, 17, WHITE, True, 1)
        if i < 3:
            line(slide, xs[i] + 6100, lane_y[0] + 800, xs[i + 1] - 360, lane_y[0] + 800, MUTED, 70)

    attacks = [
        ("Token edits", "substitution\nrearrangement\ninsertion\ndeletion"),
        ("Filters", "protected tokens\nRDKit validity\nplausibility"),
        ("Candidate pool", "Phase 2B: 651\nsets identical"),
    ]
    xs = [3000, 12400, 21800]
    for i, (a, b) in enumerate(attacks):
        box(slide, xs[i], lane_y[1], 7600, 1850, a + "\n" + b, PANEL_2, BLUE, 16, WHITE, True, 1)
        if i < 2:
            line(slide, xs[i] + 7600, lane_y[1] + 920, xs[i + 1] - 360, lane_y[1] + 920, MUTED, 70)

    models = [
        ("Baseline", "Transformer\nregressor"),
        ("Defense", "Adversarial\ntraining"),
        ("Predictions", "Tg drift\nin Kelvin"),
    ]
    for i, (a, b) in enumerate(models):
        box(slide, xs[i], lane_y[2], 7600, 1700, a + "\n" + b, PANEL, AMBER, 17, WHITE, True, 1)
        if i < 2:
            line(slide, xs[i] + 7600, lane_y[2] + 850, xs[i + 1] - 360, lane_y[2] + 850, MUTED, 70)

    audits = [
        ("D1/D2", "masking +\nposition probes"),
        ("D3", "length-only\nbaseline"),
        ("D4-D7", "length control\n5 seeds"),
        ("D6", "chemical\nseverity audit"),
    ]
    xs = [1650, 9000, 16350, 23700]
    for i, (a, b) in enumerate(audits):
        box(slide, xs[i], lane_y[3], 6100, 1600, a + "\n" + b, 0x241B1D, RED if i > 0 else AMBER, 17, WHITE, True, 1)
    line(slide, 28300, lane_y[3] - 400, 28300, lane_y[0] - 500, RED, 55)
    line(slide, 28300, lane_y[0] - 500, 13400, lane_y[0] - 500, RED, 55)
    box(slide, 13200, 2520, 10400, 620, "claim-revision feedback loop", 0x241B1D, RED, 13, WHITE, True, 1)


def model_flow(slide):
    label(slide, 1300, 3300, 5000, "implemented forward pass", CYAN)
    left_x = 1450
    steps = [
        ("PSMILES string", "raw polymer sequence"),
        ("Regex tokenizer", "45-token vocabulary"),
        ("Token ids [B,L]", "PAD id = 0"),
        ("Token embedding", "46 x 64"),
        ("Position embedding", "256 x 64 learned absolute"),
    ]
    y = 3900
    for i, (a, b) in enumerate(steps):
        box(slide, left_x, y + i * 1350, 7800, 920, a + "\n" + b, PANEL, CYAN if i < 3 else AMBER, 15, WHITE, True, 1)
        if i < len(steps) - 1:
            line(slide, left_x + 3900, y + i * 1350 + 920, left_x + 3900, y + (i + 1) * 1350 - 80, MUTED, 55)

    add_shape(slide, "com.sun.star.drawing.RectangleShape", 11200, 3800, 9000, 6200, 0x0A1726, AMBER)
    label(slide, 11800, 4100, 4600, "encoder block x2", AMBER)
    block_items = [
        ("Multi-head attention", "4 heads, d_model=64"),
        ("Add + LayerNorm", "post-norm"),
        ("Feed-forward", "64 -> 128 -> 64, ReLU"),
        ("Dropout", "0.1 inside encoder"),
    ]
    for i, (a, b) in enumerate(block_items):
        box(slide, 12000, 4750 + i * 1200, 7400, 760, a + "  |  " + b, PANEL_2, LINE, 14, WHITE, True, 1)

    right_x = 22000
    for i, (a, b, col) in enumerate([
        ("Masked mean pool", "non-PAD tokens only\nsingle 64d vector", RED),
        ("Linear head", "64 -> 1\nno head dropout", AMBER),
        ("Inverse scaler", "Tg prediction\nKelvin", GREEN),
    ]):
        box(slide, right_x, 4450 + i * 1800, 8150, 1200, a + "\n" + b, PANEL, col, 16, WHITE, True, 1)
        if i < 2:
            line(slide, right_x + 4075, 5650 + i * 1800, right_x + 4075, 6250 + i * 1800, MUTED, 55)

    line(slide, 9250, 6600, 11200, 6600, MUTED, 70)
    line(slide, 20200, 6900, 22000, 6900, MUTED, 70)
    metric_card(slide, 1600, 12600, 9000, 2900, "86,337", "parameters", RED, "204 Tg training samples")
    metric_card(slide, 12150, 12600, 9000, 2900, "1/L", "mean-pool dilution", AMBER, "confirmed by drift correlation")
    metric_card(slide, 22700, 12600, 7400, 2900, "No CLS", "active pooling path is mean", CYAN, "CLS code exists but unused")


def bar(slide, x, y, label, value, maxv, color):
    box(slide, x, y, 5200, 560, label, NAVY, NAVY, 15, WHITE)
    add_shape(slide, "com.sun.star.drawing.RectangleShape", x+5400, y+90, 6200, 320, 0x12283B, 0x12283B)
    add_shape(slide, "com.sun.star.drawing.RectangleShape", x+5400, y+90, int(6200*value/maxv), 320, color, color)
    box(slide, x+11800, y-40, 1800, 520, f"{value:.2f}", NAVY, NAVY, 15, WHITE, True)


def make_deck():
    global CURRENT_DOC
    proc, desktop = connect()
    doc = desktop.loadComponentFromURL("private:factory/simpress", "_blank", 0, ())
    CURRENT_DOC = doc
    doc.DrawPages.remove(doc.DrawPages.getByIndex(0))
    doc.DrawPages.getByIndex if False else None
    doc.Presentation.IsEndless = False
    doc.DrawPages

    s = add_slide(doc, "Materials Adversarial Forensic Review", "A clean research deck built from the actual repository evidence")
    box(s, 1450, 4300, 19600, 3100, "The audit changes the story.", NAVY, NAVY, 42, WHITE, True)
    box(s, 1500, 7550, 16800, 3000, "The model is genuinely unstable under token-level perturbations, but the original length-change explanation does not survive forensic testing.", NAVY, NAVY, 24, MUTED)
    metric_card(s, 20200, 5000, 4700, 2700, "247", "Tg samples after cleaning", CYAN)
    metric_card(s, 25500, 5000, 4700, 2700, "n=6", "sealed test split", RED)
    metric_card(s, 20200, 8250, 4700, 2700, "86k", "Transformer parameters", AMBER)
    metric_card(s, 25500, 8250, 4700, 2700, "5", "replication seeds", GREEN)
    box(s, 1500, 12600, 28600, 1200, "Claim status: real instability confirmed | length mechanism refuted | chemistry-preserving robustness not yet demonstrated", 0x241B1D, RED, 19, WHITE, True, 1)

    section_slide(doc, "01 / Framing", "What question is this project actually answering?", "Separate real model vulnerability from artifacts introduced by representation, split design, pooling, and attack construction.")

    s = add_slide(doc, "Problem Statement", "What the project is really trying to solve")
    bullet_text(s, 1400, 3900, 14500, 9000, [
        "Can sequence models for polymers stay stable when valid-looking PSMILES strings are perturbed?",
        "Do observed prediction drifts reveal true adversarial vulnerability, or artifacts of representation, split design, pooling, and attack severity?",
        "Can adversarial training improve robustness without damaging clean prediction accuracy?",
        "Which claims are actually supported once the model, data, and attack pipeline are forensically audited?",
    ], 22)
    box(s, 17600, 4200, 12800, 6100, "Why it matters\nA materials ML system that changes Tg or bandgap predictions after small string edits can look scientifically convincing while responding to syntax, length, or candidate-selection artifacts instead of chemistry.", 0x13283A, AMBER, 23, WHITE, True)

    s = add_slide(doc, "Project Scope And Data", "Two related project threads; Tg forensic audit is the current deck focus")
    bullet_text(s, 1300, 3600, 14500, 9500, [
        f"Bandgap milestone: {bandgap['n_train']}/{bandgap['n_val']}/{bandgap['n_test']} split, clean test MAE {bandgap['test_mae']:.4f} eV, R² {bandgap['test_r2']:.3f}.",
        "Tg forensic audit: final polymer properties table has 741 rows; Tg non-null = 443.",
        "After RDKit canonicalization and conflicting duplicate removal: 247 Tg samples.",
        "Scaffold split: train 204 / validation 37 / sealed test 6.",
    ], 20)
    box(s, 17600, 4500, 12500, 5400, "Important deck choice\nThe forensic findings supersede the earlier length-mechanism narrative, so this PPT presents the audit as the main result and uses older claims only as background.", 0x271A1A, RED, 22, WHITE, True)

    s = add_slide(doc, "Research Questions", "The questions this project can and cannot answer")
    for i, q in enumerate([
        "RQ1: Are polymer sequence regressors vulnerable to valid token-space perturbations?",
        "RQ2: Is the vulnerability specifically caused by length-changing attacks?",
        "RQ3: Does adversarial training generalize across attack families?",
        "RQ4: Are the attack labels chemically justified as property-preserving?",
        "RQ5: What gaps must be fixed before publishing a strong robustness claim?",
    ]):
        box(s, 1600, 3600 + i*2100, 29500, 1350, q, 0x102B42, CYAN if i in (0,4) else AMBER, 22, WHITE, True)

    s = add_slide(doc, "Literature Review And Gap", "Positioning, without pretending this project solved physics")
    bullet_text(s, 1300, 3500, 14200, 10000, [
        "Polymer property prediction often uses string representations such as SMILES/PSMILES because they are easy to tokenize.",
        "Transformers can model token context, but small-data polymer datasets make shortcut learning plausible.",
        "Standard reports emphasize clean MAE/R² more than invariance under equivalent or near-equivalent representations.",
        "Prior adversarial examples in chemistry often conflate validity, plausibility, and unchanged target labels.",
    ], 19)
    box(s, 17200, 3900, 13000, 7800, "Exact gap this work exposes\nThe missing experiment is not more attacks. It is a clean, representation-level perturbation where the molecule and label are provably unchanged, plus matched-budget evaluation and cross-validated baselines.", 0x13283A, AMBER, 24, WHITE, True)

    section_slide(doc, "02 / Architecture", "The actual system is an audit loop, not just a model.", "Data processing, token attacks, model prediction, and forensic diagnostics all determine what claims are valid.")

    s = add_slide(doc, "System Architecture", "Generated from the implemented pipeline")
    data_flow(s)

    s = add_slide(doc, "Model Architecture", "Generated from transformer.py, regression.py, and model.yaml")
    model_flow(s)

    s = add_slide(doc, "Attack And Evaluation Pipeline", "Valid candidates only, but validity is not the same as label preservation")
    bullet_text(s, 1200, 3600, 14000, 10200, [
        "Attack families: substitution, rearrangement, insertion, deletion, randomization/MCMC infrastructure.",
        "Candidates pass token-role protections, RDKit parsing, plausibility filters, and prediction drift scoring.",
        "Phase 2B symmetrical evaluation used identical candidate sets for baseline and defended models.",
        "Threshold in Tg audit: 52.02 K, equal to Phase 1 test MAE, but the test set has only 6 samples.",
    ], 19)
    for i, (k, v) in enumerate([("candidate sets identical", "true"), ("Phase 2B candidates", "651"), ("unique candidates", "651"), ("overlap with old Phase 1E", "21.97%")]):
        box(s, 17400, 3800+i*1850, 10400, 1150, f"{k}\n{v}", 0x102B42, GREEN if i==0 else CYAN, 19, WHITE, True)

    section_slide(doc, "03 / Evidence", "The strongest results are forensic, not promotional.", "The useful finding is not that every early claim was right. It is that the audit identifies exactly which claims survive.")

    s = add_slide(doc, "Baseline Findings", "The model is unstable, but clean performance is fragile")
    c = clean["phase1_baseline"]
    bullet_text(s, 1300, 3500, 14200, 9300, [
        f"Tg Phase 1 baseline validation MAE: {c['val']['mae']:.2f} K; test MAE: {c['test']['mae']:.2f} K.",
        f"Validation R²: {c['val']['r2']:.3f}; test R²: {c['test']['r2']:.3f}.",
        "The negative R² values mean clean predictive claims are weak in this Tg split.",
        "The model still exhibits real prediction drift under token perturbations.",
    ], 20)
    box(s, 17400, 4300, 12000, 5000, "Interpretation\nThe audit preserves the vulnerability finding but downgrades the explanation: unstable model behavior is real; the causal story was wrong.", 0x13283A, AMBER, 23, WHITE, True)

    s = add_slide(doc, "Forensic Result 1: Length Claim Refuted", "Length-preserving controls drift as much as length-changing edits")
    d4 = diag["D4_D7_order_vs_length"]
    means = {k: sum(v[k] for v in d4.values())/len(d4) for k in ["shuffle", "reverse", "duplicate_plus1"]}
    for i, k in enumerate(["shuffle", "reverse", "duplicate_plus1"]):
        bar(s, 2200, 4300+i*1500, k, means[k], 42, RED if k!="duplicate_plus1" else AMBER)
    box(s, 17500, 4500, 11800, 5200, "Audit conclusion\nShuffle and reverse preserve length and token multiset, yet produce 30-32 K mean drift across five seeds. The model is not specifically length-sensitive.", 0x271A1A, RED, 23, WHITE, True)

    s = add_slide(doc, "Forensic Result 2: Pooling Dilution", "Drift scales with 1/L, exactly what mean pooling predicts")
    bins = diag["D5_mean_pool_dilution"]["by_length_bin"]
    y = 3900
    for label, vals in bins.items():
        bar(s, 1400, y, f"{label} insertion", vals["insertion"], 80, AMBER); y += 1000
        bar(s, 1400, y, f"{label} deletion", vals["deletion"], 80, CYAN); y += 1250
    box(s, 18700, 4300, 10200, 4700, f"Correlations\nlength vs insertion drift: {diag['D5_mean_pool_dilution']['corr_len_insertion_drift']:.3f}\n1/length vs insertion drift: {diag['D5_mean_pool_dilution']['corr_invlen_insertion_drift']:.3f}", 0x102B42, AMBER, 24, WHITE, True)

    s = add_slide(doc, "Forensic Result 3: Attacks Are Unequal", "Attack-family comparisons confound mechanism with chemical severity")
    fam = diag["D6_chemical_destructiveness"]
    y = 3700
    for name, color in [("substitution", CYAN), ("insertion", AMBER), ("deletion", AMBER), ("rearrangement", GREEN)]:
        v = fam[name]
        box(s, 1300, y, 29200, 1050, f"{name}: n={v['n_valid']} | validity={100*v['validity_rate']:.1f}% | mean ΔMW={v['mean_abs_dMW']:.2f} | formula preserved={100*v['formula_preserved_frac']:.1f}% | mean drift={v['mean_abs_drift']:.2f} K", 0x102B42, color, 17, WHITE, True)
        y += 1350
    box(s, 4200, 9600, 23600, 1600, "Rearrangement preserves formula in 100% of valid cases; insertion/deletion change heavy atoms. These are not matched experimental treatments.", 0x271A1A, RED, 21, WHITE, True, 1)

    s = add_slide(doc, "Forensic Result 4: Controls Beat The Transformer", "The model has not demonstrably learned chemistry in the Tg audit")
    for i, (name, vals, color) in enumerate([
        ("Mean predictor val MAE", diag["D3_controls"]["mean"][0], MUTED),
        ("Length-only val MAE", diag["D3_controls"]["length_only"][0], GREEN),
        ("Bag-of-tokens val MAE", diag["D3_controls"]["bag_of_tokens"][0], CYAN),
        ("Transformer val MAE", diag["D3_controls"]["transformer"][0], AMBER),
    ]):
        bar(s, 1900, 4100+i*1350, name, vals, 95, color)
    box(s, 17600, 4200, 11500, 4700, "The uncomfortable result\nA one-feature length-only regression beats the Transformer on validation: 76.77 K vs 79.04 K.", 0x271A1A, RED, 23, WHITE, True)

    section_slide(doc, "04 / Defense", "Adversarial training helped locally, then hit its limits.", "The deck states both sides: controlled improvements exist, but general chemical invariance is not established.")

    s = add_slide(doc, "Defended Model: What Worked", "Local robustness improved in Phase 2B under controlled candidate sets")
    for i, name in enumerate(["substitution", "insertion", "deletion", "rearrangement"]):
        b = phase2b["families"][name]["baseline"]["mean"]
        d = phase2b["families"][name]["phase2b_controlled"]["mean"]
        box(s, 1300, 3600+i*1700, 13300, 1000, f"{name} baseline mean drift: {b:.2f} K", 0x102B42, CYAN, 17, WHITE, True)
        box(s, 15400, 3600+i*1700, 13300, 1000, f"defended: {d:.2f} K", 0x102B42, GREEN if d < b else AMBER, 17, WHITE, True)
    box(s, 4200, 11200, 23600, 1500, f"Pooled Phase 2B mean drift dropped by {phase2b['paired_pooled']['mean_reduction']:.2f} K; {100*phase2b['paired_pooled']['fraction_improved']:.1f}% of paired candidates improved.", 0x13283A, GREEN, 21, WHITE, True, 1)

    s = add_slide(doc, "Defended Model: What Failed", "Replication and label assumptions weaken the defense claim")
    bullet_text(s, 1300, 3500, 14500, 9200, [
        "Phase 2C used five seeds: 20260815-20260819.",
        "Clean evaluation remained validation-only for selection; manifest says the test set was never loaded.",
        "The apparent Phase 2A improvement was mostly scaler mismatch; only 23.7% survived controlled decoding.",
        "No current attack family is a valid representation-level label-preserving attack.",
    ], 19)
    box(s, 17600, 4200, 11600, 5000, "Bottom line\nAdversarial training may reduce seen-family drift locally, but it does not yet prove chemical invariance or general robust learning.", 0x271A1A, RED, 23, WHITE, True)

    section_slide(doc, "05 / Conclusions", "The final answer is sharper because it is narrower.", "Publish the honest claim: instability is real; the length story and label-preservation assumption need repair.")

    s = add_slide(doc, "Research Questions Answered", "Exact answer from the audit")
    answers = [
        ("RQ1", "Yes, model instability exists: 20-40 K drift under token perturbation."),
        ("RQ2", "No. Length is not the mechanism; length-preserving shuffle/reverse are equally disruptive."),
        ("RQ3", "Only partly. Seen-family/local improvements exist, but generalization is not established."),
        ("RQ4", "No. Current attack families change chemistry or isomerize; none is a clean representation-level attack."),
        ("RQ5", "Need SMILES randomization, matched budgets, CV baselines, and pooling/position ablations."),
    ]
    for i, (rq, ans) in enumerate(answers):
        box(s, 1400, 3450+i*1850, 3200, 1150, rq, GREEN if i==0 else (RED if i in (1,3) else AMBER), GREEN if i==0 else (RED if i in (1,3) else AMBER), 24, 0x061019, True, 1)
        box(s, 5000, 3450+i*1850, 25200, 1150, ans, 0x102B42, LINE, 19, WHITE, True)

    s = add_slide(doc, "Limitations And Exact Gaps", "Drawbacks stated plainly")
    bullet_text(s, 1300, 3500, 29000, 9700, [
        "Tg test set has n=6, so clean-MAE comparisons are anecdotal.",
        "Scaffold split is confounded: train/val/test mean Tg = 330.7/384.5/463.2 K; mean token length = 22.6/43.5/56.7.",
        "Validity-conditioned drift is a survivorship filter; validity rates differ by family.",
        "Attack budgets are not matched, and max drift scales with candidate count.",
        "No ab initio, synthesis, or experimental validation confirms adversarial candidates preserve the true property.",
        "No implemented representation-level attack yet; SMILES randomization is the missing control.",
    ], 20)

    s = add_slide(doc, "Novelty", "What is actually new and defensible")
    bullet_text(s, 1300, 3600, 29200, 9800, [
        "A forensic audit that separates model instability from a false length-change mechanism.",
        "Direct architectural diagnosis tying drift to learned positional embeddings, masked mean pooling, and small-data training.",
        "Chemical-severity audit showing why attack-family comparisons were not controlled experiments.",
        "Replication across five seeds showing Phase 1F's causal story does not hold.",
        "A revised research program: representation-level SMILES randomization before bigger attacks or new architectures.",
    ], 21)

    s = add_slide(doc, "Next Slide / Next Work", "Use this as the closing action slide")
    for i, (n, text, col) in enumerate([
        ("1", "SMILES-randomization invariance test: same molecule, provably same label.", GREEN),
        ("2", "Positional-encoding ablation across 5 seeds.", CYAN),
        ("3", "k-fold scaffold CV: length-only, bag-of-tokens, Transformer.", AMBER),
        ("4", "Budget-matched attacks with median drift, validity rate, and no-op counts.", AMBER),
        ("5", "Only then consider graph models, MCMC, or broader defenses.", RED),
    ]):
        box(s, 1600, 3400+i*2050, 2500, 1250, n, col, col, 28, 0x061019, True, 1)
        box(s, 4500, 3400+i*2050, 26000, 1250, text, 0x102B42, LINE, 21, WHITE, True)

    s = add_slide(doc, "Takeaway", "Best design, honest claims")
    box(s, 2400, 5000, 28400, 4700, "The strongest version of this project is not: 'we proved length-changing attacks break polymer Transformers.'\n\nIt is: 'we built and audited a materials adversarial framework, found real token-level instability, and showed that several attractive robustness claims collapse unless representation-level invariance, matched attack severity, and small-data baselines are handled first.'", 0x102B42, AMBER, 25, WHITE, True, 1)

    for p in [TMP_ODP, OUT]:
        if p.exists():
            p.unlink()
    url = uno.systemPathToFileUrl(str(TMP_ODP))
    doc.storeAsURL(url, (prop("FilterName", "impress8"),))
    doc.storeAsURL(uno.systemPathToFileUrl(str(OUT)), (prop("FilterName", "Impress MS PowerPoint 2007 XML"),))
    doc.close(True)
    proc.terminate()


if __name__ == "__main__":
    make_deck()
    print(OUT)

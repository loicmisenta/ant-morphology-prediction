import os
import re
import csv
import math
import cv2
import numpy as np
from collections import defaultdict, Counter
import glob
import pandas as pd

# =========================================================
# CONFIG: set the image folder, the output file, and the lists if needed
# =========================================================
IMAGE_FOLDER = 'dossier_images/original'

# One CSV per collection goes here
RESULTS_FOLDER = 'scale_results'
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# One sub-folder of check pictures per collection goes here
DEBUG_ROOT = 'bar_check'
DEBUG_FOLDER = DEBUG_ROOT          # changed for each collection in the main loop
os.makedirs(DEBUG_ROOT, exist_ok=True)

# Run only these collections (None -> all collections)
ANT_LIST = [
    'anic32', 'antweb', 'casent', 'castype', 'dzup', 'ecofog', 'fmnhins',
    'focol', 'icn', 'inb', 'inbiocri', 'jdm32', 'jtlc', 'kg', 'lacment',
    'mcz', 'nbairhym', 'ncbs', 'nhmuk', 'sam', 'thnhm', 'ufv', 'usnment',
    'utic', 'zrc', 'asusibr', 'blf', 'cocy'
]
PATTERN_COLLECTION = r"^(" + "|".join(sorted(ANT_LIST, key=len, reverse=True)) + r")"
REFINED_LIST = None
ONLY_COLLECTIONS = None

# =========================================================
# Scale bar measurement (thin horizontal line method)
# =========================================================
TEXT_SEARCH_MARGIN_X = 600      # horizontal search margin around the text
TEXT_SEARCH_MARGIN_Y = 30       # vertical search margin around the text
BAR_CONTRAST = 30               # min. difference with the LOCAL background
BAR_MIN_WIDTH_PX = 40           # final minimum bar length
BAR_MAX_THICKNESS_PX = 16       # thicker blobs (legs, body) are erased
BAR_MIN_ASPECT = 8.0            # length / thickness
BAR_SEGMENT_PX = 20             # shortest horizontal piece kept before merging
BAR_GAP_TOLERANCE_PX = 15       # gaps bridged (leg crossing the bar, noise)


# EasyOCR reader: reuses an existing `reader` if defined, otherwise creates one
try:
    reader
except NameError:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=True)


# =========================================================
# SHARED CONSTANTS AND HELPERS
# =========================================================
ALLOWLIST = '0123456789.,mMµμuUnNcCpPrRIl|oOQD '
UNIT_TO_MM = {'mm': 1.0, 'µm': 1e-3, 'nm': 1e-6, 'cm': 10.0}
CORRECTED_UNITS = {'um': 'µm', 'μm': 'µm', 'uum': 'µm', 'pm': 'µm',
                   'mn': 'mm', 'rn': 'mm', 'min': 'mm',
                   'rr': 'mm', 'rm': 'mm', 'rnm': 'mm'}
UNIT_REGEX = r'^(.*?)\s*(µm|μm|uum|um|mm|nm|cm|pm|mn|rnm|rn|rr|rm|min)$'

# Plausible real width of an image, in mm (used to reject impossible readings)
MIN_WIDTH_MM, MAX_WIDTH_MM = 0.1, 50

CSV_COLUMNS = ['image', 'number', 'unit', 'bar_px', 'px_per_mm', 'mm_per_px', 'comments']


def normalize_unit(unit):
    """'um', 'UM', 'uum', 'pm' -> 'µm'; 'MM' -> 'mm'."""
    u = unit.lower()
    return CORRECTED_UNITS.get(u, u)


def join_comments(*parts):
    return "; ".join(p for p in parts if p)


def box_to_rect(box):
    """EasyOCR box (4 corner points) -> (xmin, xmax, ymin, ymax)."""
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))


def merge_rects(rects):
    return (min(r[0] for r in rects), max(r[1] for r in rects),
            min(r[2] for r in rects), max(r[3] for r in rects))


def image_width_mm(W, bar_px, number, unit):
    """Real width of the image (in mm) implied by a reading of the scale bar."""
    real_length_mm = number * UNIT_TO_MM.get(unit, 1.0)
    if number <= 0 or real_length_mm <= 0:
        return float('inf')
    return W / (bar_px / real_length_mm)


# =========================================================
# FIRST READING (original method)
# OCR on the whole image, then three refinements:
#   Alt 1: full block "number + unit" (uum and pm are converted to um)
#   Alt 2: number and unit read on different lines -> combined
#   Alt 3: unit read without number -> number assumed to be 1
# Also returns the position of the text, used later to find the scale bar.
# =========================================================
pattern_strict = r"([0-9]+[.,]?[0-9]*)\s*(mm|µm|um|uum|nm|cm|pm)"
pattern_number = r"^([0-9]+[.,]?[0-9]*)$"
pattern_unit = r"^(mm|µm|um|uum|nm|cm|pm)$"
pattern_error_one = r"^[Il1]\s*(mm|µm|um|uum|nm|cm|pm)$"


def first_read(image_path):
    """Returns ((number, unit, comment, text_rect) or None, texts_read)."""
    results = reader.readtext(image_path)
    read_text = [res[1].strip() for res in results]

    filtered = []                            # list of (text, rect)
    for res in results:
        rect = box_to_rect(res[0])
        t_clean = res[1].strip()
        t_lower = t_clean.lower().replace(" ", "")
        if t_lower.endswith('pm') or t_lower.endswith('uum'):
            t_clean = t_clean.lower().replace('pm', 'µm').replace('uum', 'µm')
        if t_lower in ['min', 'amm', 'tmm', 'lomm', '~mm', 'mm']:
            t_clean = "1 mm"
        elif '0.s' in t_lower:
            t_clean = "0.5 mm"
        filtered.append((t_clean, rect))

    number, unit = None, None
    comment = ""
    glued = False                            # number and unit came from separate fragments
    used_rects = []

    # Alt 1: full block
    for text, rect in filtered:
        match = re.search(pattern_strict, text, re.IGNORECASE)
        if match:
            number = float(match.group(1).replace(',', '.'))
            unit = match.group(2)
            if unit.lower() in ('uum', 'pm'):
                unit = 'µm'
                comment = "unit corrected to um"
            return (number, unit, comment, rect), read_text

    # Alt 2: number and unit on separate lines
    for text, rect in filtered:
        match_num = re.match(pattern_number, text)
        if match_num:
            number = float(match_num.group(1).replace(',', '.'))
            glued = True
            used_rects.append(rect)
        match_un = re.match(pattern_unit, text, re.IGNORECASE)
        if match_un:
            unit = match_un.group(1)
            if unit.lower() in ('uum', 'pm'):
                unit = 'µm'
            glued = True
            used_rects.append(rect)
        match_corr = re.match(pattern_error_one, text, re.IGNORECASE)
        if match_corr:
            first_char = text.strip()[0]
            number = 1.0
            unit = match_corr.group(1)
            if unit.lower() in ('uum', 'pm'):
                unit = 'µm'
            if first_char != '1':            # a real misread (I, l, i, L)
                comment = f"'{first_char}' corrected to 1"
            used_rects.append(rect)
    if number is not None and unit is not None:
        if not comment and glued:
            comment = "number and unit read on separate lines, combined"
        return (number, unit, comment, merge_rects(used_rects)), read_text

    # Alt 3: unit only
    for text, rect in filtered:
        match_unit_only = re.match(pattern_unit, text, re.IGNORECASE)
        if match_unit_only:
            unit = match_unit_only.group(1)
            if unit.lower() in ('uum', 'pm'):
                unit = 'µm'
            return (1.0, unit, "only unit read, number assumed to be 1", rect), read_text

    return None, read_text


# =========================================================
# SECOND READING (fallback): voting algorithm
# The scale bar is detected and the legend below it is cropped.
# The crop is enlarged (x3, x4, x6) and preprocessed in 7 ways
# (plain grey, CLAHE contrast, white-text and dark-text extraction,
# Otsu threshold and its inverse, top-hat). Each of the 21 versions is read
# by OCR and votes for a value; the best-scoring value wins.
# =========================================================
def detect_scale_bar(gray, bottom_band=0.25):
    """Used only to crop the legend for the vote fallback.
    Looks for a long, thin horizontal line in the bottom part of the image.
    Returns (x_start, x_end, y) in full-image coordinates, or None."""
    H, W = gray.shape
    y_off = int(H * (1 - bottom_band))
    roi = gray[y_off:, :]
    med = np.median(roi)
    candidates = []
    for polarity in ('white', 'black'):
        if polarity == 'white':
            mask = (roi > med + 30).astype(np.uint8) * 255
        else:
            mask = (roi < med - 30).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, W // 40), 1))
        horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        n, _, stats, _ = cv2.connectedComponentsWithStats(horizontal)
        for i in range(1, n):
            x, y, w, h, _area = stats[i]
            if w >= 40 and h <= 8 and w / max(h, 1) > 15:
                candidates.append((int(w), int(x), int(x + w), int(y_off + y + h // 2)))
    if not candidates:
        return None
    _, x0, x1, y = max(candidates)
    return int(x0), int(x1), int(y)


def crop_legend(img_bgr):
    """Crops the zone under the scale bar where the legend text sits.
    Returns (cropped grayscale image, info dict)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    bar = detect_scale_bar(gray)

    if bar:
        x0, x1, y = bar
        margin = max(40, (x1 - x0) // 3)
        y_start, y_end = y + 2, min(H, y + 45)
        x_start, x_end = max(0, x0 - margin), min(W, x1 + margin)
        info = {'bar_px': x1 - x0}
    else:
        # No bar found: bottom 10% of the image, centered
        y_start, y_end = int(H * 0.90), H
        x_start, x_end = int(W * 0.15), int(W * 0.85)
        info = {'bar_px': None}

    return gray[y_start:y_end, x_start:x_end], info


def make_variants(crop_gray):
    """Builds 21 versions of the crop (3 sizes x 7 preprocessings)."""
    out = {}
    for factor in (3, 4, 6):
        g = cv2.resize(crop_gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
        g = cv2.copyMakeBorder(g, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
        f = f'x{factor}_'
        out[f + 'gray'] = g
        out[f + 'clahe'] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(g)
        med = np.median(g)
        out[f + 'white_inv'] = 255 - (g > med + 25).astype(np.uint8) * 255
        out[f + 'dark_inv'] = 255 - (g < med - 40).astype(np.uint8) * 255
        _, otsu = cv2.threshold(cv2.GaussianBlur(g, (3, 3), 0), 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        out[f + 'otsu'] = otsu
        out[f + 'otsu_inv'] = 255 - otsu
        th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
        out[f + 'tophat_inv'] = 255 - cv2.normalize(th, None, 0, 255, cv2.NORM_MINMAX)
    return out


def scale_candidates(texts):
    """Turns OCR text fragments into possible (number, unit, comment) readings,
    from most to least likely. An ambiguous reading can return several candidates."""
    s = re.sub(r'\s+', ' ', ' '.join(texts).lower().strip())
    m = re.match(UNIT_REGEX, s)
    if not m:
        return []
    num_txt, unit = m.groups()
    unit = CORRECTED_UNITS.get(unit, unit)

    # Letters that look like digits, fixed in the number part only
    num_txt = num_txt.translate(str.maketrans({
        'o': '0', 'u': '0', 'c': '0', 'd': '0', 'q': '0',
        'l': '1', 'i': '1', '|': '1', 'r': '1', ',': '.'}))
    num_txt = re.sub(r'[^0-9. ]', '', num_txt).strip()

    # No digit read at all: unit only, number assumed to be 1 (weak reading)
    if not re.search(r'\d', num_txt):
        return [(1.0, unit, 'digit not read, number assumed to be 1')]

    tokens = num_txt.split()
    raw_numbers = []
    if len(tokens) >= 2 and all(t.isdigit() for t in tokens):
        raw_numbers.append(tokens[0] + '.' + ''.join(tokens[1:]))   # '1 00' -> '1.00'
        raw_numbers.append(''.join(tokens))                         # '1 00' -> '100'
    else:
        t = ''.join(tokens)
        raw_numbers.append(t)
        if '.' not in t and len(t) >= 3:                            # '100' -> also '1.00'
            raw_numbers.append(t[0] + '.' + t[1:])

    out = []
    for raw in raw_numbers:
        try:
            value = float(raw)
            if value > 0:                   # a zero is never a valid scale
                out.append((value, unit, ''))
        except ValueError:
            pass
    return out


def prior_weight(number):
    """Real scale bars are usually {1, 2, 2.5, 5} x 10^n. Penalises other values."""
    if number <= 0:
        return 0.1
    mantissa = number / 10 ** math.floor(math.log10(number))
    return 1.0 if any(abs(mantissa - x) < 1e-6 for x in (1, 2, 2.5, 5)) else 0.4


def read_scale_vote(img_bgr):
    """Fallback reading by voting over preprocessed versions of the legend.
    Returns ((number, unit, comment) or None, texts_read)."""
    W = img_bgr.shape[1]
    crop, info = crop_legend(img_bgr)

    votes, all_texts = Counter(), []
    weak = {}                                # candidate -> comment of a weak reading
    for variant_name, variant in make_variants(crop).items():
        res = reader.readtext(variant, allowlist=ALLOWLIST, detail=1, paragraph=False)
        res = sorted(res, key=lambda r: r[0][0][0])          # left to right
        texts = [r[1] for r in res]
        if texts:
            all_texts.append(' '.join(texts))
        conf = float(np.mean([r[2] for r in res])) if res else 0.0

        for number, unit, comment in scale_candidates(texts):
            if info['bar_px']:
                width = image_width_mm(W, info['bar_px'], number, unit)
                if not (MIN_WIDTH_MM <= width <= MAX_WIDTH_MM):
                    continue                                  # physically impossible
            votes[(number, unit)] += conf * prior_weight(number) * (0.3 if comment else 1.0)
            if comment:
                weak[(number, unit)] = comment
            break

    if not votes:
        return None, sorted(set(all_texts))

    ranking = votes.most_common()
    (number, unit), score = ranking[0]

    parts = ['vote fallback']
    if (number, unit) in weak:
        parts.append(weak[(number, unit)])
    if len(ranking) > 1 and ranking[1][1] > 0.8 * score:
        parts.append("close result, please check")
    return (number, unit, "; ".join(parts)), all_texts


# =========================================================
# SCALE BAR MEASUREMENT
# 1) local background removed (median blur) -> white / black masks
# 2) thick blobs (legs, body...) are erased from the mask
# 3) only thin horizontal lines are kept, small gaps are bridged
# 4) the best candidate (long, and close to the text) is the bar
# =========================================================
def find_bar_by_contours(gray, x_start, x_end, y_start, y_end, ref_y=None):
    """Looks for the scale bar (white, then black) in a region of the image.
    ref_y: vertical position of the text (candidates closer to it are preferred).
    Returns a dict {width_px, thickness_px, type, x, y} or None."""
    roi = gray[y_start:y_end, x_start:x_end]
    if roi.size == 0 or roi.shape[1] < BAR_MIN_WIDTH_PX:
        return None

    # local background: a thin line disappears in a median window larger than itself
    k = 2 * BAR_MAX_THICKNESS_PX + 1
    k = min(k, (min(roi.shape) // 2) * 2 - 1)
    k = max(k, 3)
    bg = cv2.medianBlur(roi, k)
    diff = roi.astype(np.int16) - bg.astype(np.int16)

    masks = [
        ((diff > BAR_CONTRAST).astype(np.uint8) * 255, 'white'),
        ((diff < -BAR_CONTRAST).astype(np.uint8) * 255, 'black'),
    ]

    se = cv2.getStructuringElement
    candidates = []
    for mask, bar_type in masks:
        # keep only horizontal pieces
        thin = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                se(cv2.MORPH_RECT, (BAR_SEGMENT_PX, 1)))

        # erase thick blobs (legs, body): they survive a square opening, a bar does not
        t = BAR_MAX_THICKNESS_PX + 1
        thick = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se(cv2.MORPH_RECT, (t, t)))
        thick = cv2.dilate(thick, se(cv2.MORPH_RECT, (7, 7)))
        thin[thick > 0] = 0

        # bridge gaps along the bar (leg crossing it, noise) ...
        thin = cv2.morphologyEx(thin, cv2.MORPH_CLOSE,
                                se(cv2.MORPH_RECT, (BAR_GAP_TOLERANCE_PX, 1)))
        # ... and join the two lines of a bar drawn as an outline (with end caps)
        thin = cv2.dilate(thin, se(cv2.MORPH_RECT, (1, 5)))

        n, _, stats, _ = cv2.connectedComponentsWithStats(thin)
        for i in range(1, n):
            x, y, w, h, _area = stats[i]
            h_real = max(1, int(h) - 4)          # remove the 5-px dilation
            if (w >= BAR_MIN_WIDTH_PX and h_real <= BAR_MAX_THICKNESS_PX
                    and w / h_real >= BAR_MIN_ASPECT):
                cy = y_start + y + h / 2
                score = w
                if ref_y is not None:
                    score = w / (1.0 + abs(cy - ref_y) / 30.0)
                candidates.append((score, int(w), h_real, bar_type,
                                   int(x), int(y) + (int(h) - h_real) // 2))

    if not candidates:
        return None
    _, w, h, bar_type, bx, by = max(candidates)
    return {'width_px': w, 'thickness_px': h, 'type': bar_type,
            'x': x_start + bx, 'y': y_start + by}


def measure_scale_bar(gray, text_rect=None):
    """Returns (bar dict or None, where it was searched)."""
    H, W = gray.shape
    if text_rect is not None:
        xmin, xmax, ymin, ymax = text_rect
        bar = find_bar_by_contours(gray,
                                   max(0, xmin - TEXT_SEARCH_MARGIN_X),
                                   min(W, xmax + TEXT_SEARCH_MARGIN_X),
                                   max(0, ymin - TEXT_SEARCH_MARGIN_Y),
                                   min(H, ymax + TEXT_SEARCH_MARGIN_Y),
                                   ref_y=(ymin + ymax) / 2)
        if bar:
            return bar, 'near text'
    bar = find_bar_by_contours(gray, 0, W, int(H * 0.75), H)
    if bar:
        return bar, 'bottom band'
    return None, None

# =========================================================
# CHECK PICTURES
# For each image: the full picture with the detected bar boxed in red and
# the OCR text boxed in blue, plus a 2x zoom around the bar.
# =========================================================
def save_bar_check(img_bgr, filename, bar, number, unit, text_rect):
    stem = os.path.splitext(filename)[0]
    vis = img_bgr.copy()
    H, W = vis.shape[:2]

    if text_rect is not None:
        xmin, xmax, ymin, ymax = text_rect
        cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)   # blue = text

    if bar:
        x, y, w, h = bar['x'], bar['y'], bar['width_px'], bar['thickness_px']
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)        # red = bar
        label = f"{number} {unit} = {w}px"
        cv2.putText(vis, label, (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # zoomed crop around the bar
        pad = 60
        crop = vis[max(0, y - pad):min(H, y + h + pad),
                   max(0, x - pad):min(W, x + w + pad)]
        if crop.size:
            crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(DEBUG_FOLDER, stem + '_zoom.jpg'), crop)

    cv2.imwrite(os.path.join(DEBUG_FOLDER, stem + '_check.jpg'), vis)


# =========================================================
# PROCESS ONE IMAGE
#   1) original method (text + position of the text)
#   2) vote fallback if the original failed, crashed or returned 0
#   3) scale bar measurement around the text, then conversion
#   4) check pictures saved
# =========================================================
def process_image(image_path, refined_enabled):
    row = {c: '' for c in CSV_COLUMNS}
    img = cv2.imread(image_path)
    if img is None:
        row['comments'] = 'FAILED; image could not be loaded'
        return row
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    W = gray.shape[1]

    result, text_rect = None, None
    texts_read, fallback_reason, vote_error = [], '', ''

    # 1) Original method first
    try:
        original, texts_read = first_read(image_path)
        if original is None:
            fallback_reason = 'original method found nothing'
        elif original[0] <= 0:
            fallback_reason = f'original method read {original[0]}'
        else:
            number, unit, comment, text_rect = original
            result = (number, unit, comment)
    except Exception as e:
        fallback_reason = f'original method crashed: {e}'

    # 2) Vote fallback only if the original failed, crashed or returned 0
    if result is None and refined_enabled:
        try:
            vote_result, vote_texts = read_scale_vote(img)
            texts_read = texts_read or vote_texts
            if vote_result and vote_result[0] > 0:
                number, unit, comment = vote_result
                result = (number, unit, join_comments(comment, f"({fallback_reason})"))
                text_rect = None             # text position unknown for the vote
        except Exception as e:
            vote_error = f'vote fallback crashed: {e}'

    if result is None:
        row['comments'] = join_comments('FAILED', fallback_reason, vote_error,
                                        f'texts read: {texts_read}')
        return row

    number, unit, comment = result
    unit = normalize_unit(unit)
    comments = [comment]

    # 3) Measure the scale bar and compute the conversion
    bar, where = measure_scale_bar(gray, text_rect)
    if bar and unit in UNIT_TO_MM:
        bar_px = bar['width_px']
        length_mm = number * UNIT_TO_MM[unit]
        row['bar_px'] = bar_px
        row['px_per_mm'] = f"{bar_px / length_mm:.3f}"
        row['mm_per_px'] = f"{length_mm / bar_px:.6f}"
        comments.append(f"{bar['type']} bar found {where}, thickness {bar['thickness_px']} px")

        width = image_width_mm(W, bar_px, number, unit)
        if not (MIN_WIDTH_MM <= width <= MAX_WIDTH_MM):
            comments.append(f"value implies an image width of {width:.1f} mm, please check")
    elif bar:
        comments.append(f"bar measured ({bar['width_px']} px) but unit '{unit}' is unknown, no conversion")
        row['bar_px'] = bar['width_px']
    else:
        comments.append('scale bar not detected, no conversion')

    # 4) Save the check pictures (bar in red, text in blue)
    try:
        save_bar_check(img, os.path.basename(image_path), bar, number, unit, text_rect)
    except Exception as e:
        comments.append(f'check picture not saved: {e}')

    row['number'] = number
    row['unit'] = unit
    row['comments'] = join_comments(*comments)
    return row


# =========================================================
# GROUP IMAGES BY COLLECTION
# =========================================================
valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
all_images = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith(valid_extensions) and not f.startswith('.')
])

images_by_collection = defaultdict(list)
unknown_images = []
for filename in all_images:
    match = re.match(PATTERN_COLLECTION, filename, re.IGNORECASE)
    if match:
        images_by_collection[match.group(1).lower()].append(filename)
    else:
        unknown_images.append(filename)


# =========================================================
# MAIN LOOP: results are written to a CSV
# =========================================================
# =========================================================
# MAIN LOOP: one CSV and one check-picture folder per collection
# =========================================================
print("--- START OF GLOBAL ANALYSIS ---")

for collection_name in sorted(images_by_collection.keys()):
    if ONLY_COLLECTIONS and collection_name not in ONLY_COLLECTIONS:
        continue
    files = images_by_collection[collection_name]
    print(f"\n=== COLLECTION: {collection_name.upper()} ({len(files)} images) ===")
    refined_enabled = (REFINED_LIST is None) or (collection_name in REFINED_LIST)

    # check pictures of this collection go in their own folder
    DEBUG_FOLDER = os.path.join(DEBUG_ROOT, collection_name)
    os.makedirs(DEBUG_FOLDER, exist_ok=True)

    output_csv = os.path.join(RESULTS_FOLDER, f'scale_results_{collection_name}.csv')
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for index, filename in enumerate(files, start=1):
            row = process_image(os.path.join(IMAGE_FOLDER, filename), refined_enabled)
            row['image'] = filename
            writer.writerow(row)
            csv_file.flush()                 # keeps the results if the run is interrupted

            if row['number'] != '':
                conv = f"{row['px_per_mm']} px/mm" if row['px_per_mm'] else "no conversion"
                print(f"[{index:03d}] {filename} -> {row['number']} {row['unit']} | "
                      f"bar {row['bar_px'] or '-'} px | {conv} | {row['comments']}")
            else:
                print(f"[{index:03d}] {filename} -> FAILED | {row['comments']}")

    print(f"--- {collection_name}: saved to {output_csv} ---")

if unknown_images and not ONLY_COLLECTIONS:
    print(f"\n=== UNKNOWN COLLECTION ({len(unknown_images)} images) ===")
    with open(os.path.join(RESULTS_FOLDER, 'scale_results_unknown.csv'), 'w',
              newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for index, filename in enumerate(unknown_images, start=1):
            row = {c: '' for c in CSV_COLUMNS}
            row['image'] = filename
            row['comments'] = 'collection prefix not recognised, not processed'
            writer.writerow(row)
            print(f"[{index:03d}] {filename} -> not processed")

print(f"\n--- DONE. CSV files in {RESULTS_FOLDER}/ ---")
print(f"--- Check pictures in {DEBUG_ROOT}/<collection>/ ---")




RESULTS_FOLDER = 'scale_results'

# Columns checked, with a readable meaning for a missing value
COLUMNS = {
    'number':    'text not read (number)',
    'unit':      'text not read (unit)',
    'bar_px':    'bar not detected',
    'px_per_mm': 'no conversion',
    'mm_per_px': 'no conversion (mm/px)',
}

rows = []
for path in sorted(glob.glob(os.path.join(RESULTS_FOLDER, 'scale_results_*.csv'))):
    collection = os.path.basename(path)[len('scale_results_'):-len('.csv')]
    df = pd.read_csv(path, encoding='utf-8-sig')
    n = len(df)
    if n == 0:
        continue

    # a value is missing if the cell is empty / NaN
    missing = {col: df[col].isna() | (df[col].astype(str).str.strip() == '')
               for col in COLUMNS}
    stats = {'collection': collection, 'n_images': n}
    for col in COLUMNS:
        stats[f'% missing {col}'] = round(100 * missing[col].mean(), 1)
    # image with at least one missing value among the main columns
    any_missing = missing['number'] | missing['unit'] | missing['bar_px'] | missing['px_per_mm']
    stats['% any missing'] = round(100 * any_missing.mean(), 1)
    stats['_missing_count'] = {col: int(missing[col].sum()) for col in COLUMNS}
    stats['_any_count'] = int(any_missing.sum())
    rows.append(stats)

if not rows:
    print(f"No CSV found in {RESULTS_FOLDER}/")
else:
    report = pd.DataFrame(rows)

    # TOTAL row, computed from counts (not an average of percentages)
    total_n = int(report['n_images'].sum())
    total = {'collection': 'TOTAL', 'n_images': total_n}
    for col in COLUMNS:
        c = sum(r['_missing_count'][col] for r in rows)
        total[f'% missing {col}'] = round(100 * c / total_n, 1)
    total['% any missing'] = round(100 * sum(r['_any_count'] for r in rows) / total_n, 1)

    report = report.drop(columns=['_missing_count', '_any_count'])
    report = pd.concat([report, pd.DataFrame([total])], ignore_index=True)

    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', None)
    print(report.to_string(index=False))

    report.to_csv(os.path.join(RESULTS_FOLDER, 'missing_report.csv'),
                  index=False, encoding='utf-8-sig')
    print(f"\nReport saved to {RESULTS_FOLDER}/missing_report.csv")
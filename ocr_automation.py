#!/usr/bin/env python3
"""
OCR Desktop Automation Tool
----------------------------
Hotkey + click-triggered tool that OCRs a screen image and fills predefined
text boxes with extracted text. Blacklisted buttons are never clicked.

Usage:
  python ocr_automation.py               # Run automation (click image or hotkey)
  python ocr_automation.py --record      # Interactive setup — press keys to record positions
  python ocr_automation.py --calibrate   # Show live mouse coordinates
  python ocr_automation.py --help        # Show help
"""

import json
import sys
import time
import re
import argparse
import logging
import threading
from pathlib import Path

import pyautogui
import cv2
import numpy as np
import pytesseract
import keyboard
import requests

from pynput import mouse


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
LOG_PATH = BASE_DIR / "ocr_automation.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("ocr")

DEFAULT_CONFIG: dict = {
    "click_to_trigger": True,
    "ocr_image_region": {"x": 100, "y": 100, "width": 400, "height": 200},
    "text_boxes": [
        {"id": "first_name", "x": 500, "y": 600, "width": 300, "height": 30, "label": "First Name", "validation": None},
        {"id": "last_name",  "x": 500, "y": 650, "width": 300, "height": 30, "label": "Last Name", "validation": None},
    ],
    "blacklist_buttons": [
        {"id": "save_btn",   "x": 400, "y": 700, "width": 100, "height": 40},
        {"id": "submit_btn", "x": 520, "y": 700, "width": 100, "height": 40},
    ],
    "start_shortcut":         "space+ctrl",
    "stop_shortcut":          "ctrl+shift+x",
    "cancel_shortcut":        "esc",
    "learn_shortcut":         "ctrl+shift+c",
    "tesseract_path":         "",
    "ocr_lang":               "eng",
    "click_before_type":      True,
    "typing_delay":           0.015,
    "mouse_move_duration":    0.15,
    "ocr_retry_count":        1,
    "ocr_confidence_threshold": 0,
    "enable_rag":             False,
    "rag_api_url":            "http://localhost:11434/api/generate",
    "rag_model":              "llama3.2:1b",
    "rag_retry_count":        1,
    "rag_timeout":            120,
    "rag_prompt":             "Fix OCR errors in this text. Preserve all words, spacing, and capitalization. Return ONLY the corrected text.\n\n{text}",
}

if sys.platform == "win32":
    for _p in [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]:
        if Path(_p).exists():
            pytesseract.pytesseract.tesseract_cmd = _p
            break

_busy = False
_cancel = False
_mouse_listener = None
_last_ocr_text = ""
_last_filled_values: list[str] = []
_brain_path = BASE_DIR / "brain.json"


class NeuralBrain:
    """Real neural network brain that learns OCR correction patterns.
    
    Architecture: 2-layer MLP with character context window.
    - Input: one-hot encoded character context (11 chars → ~770 features)
    - Hidden: 128 neurons, ReLU
    - Output: vocab_size neurons, softmax
    
    Trains on (wrong_text → correct_text) pairs via SGD.
    Starts untrained — improves with every correction.
    Falls back to fuzzy matching when uncertain.
    """

    VOCAB = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@._-/:$,!"

    def __init__(self):
        self.vocab = sorted(set(self.VOCAB))
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(self.vocab)}

        self.context_size = 11
        self.hidden_size = 128
        input_size = self.context_size * self.vocab_size

        self.W1 = np.random.randn(input_size, self.hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.vocab_size) * np.sqrt(2.0 / self.hidden_size)
        self.b2 = np.zeros(self.vocab_size)

        self.examples: list[tuple[str, str]] = []
        self.char_examples: list[tuple[np.ndarray, int]] = []
        self.total_trained = 0
        self.pending_train = 0
        self.train_frequency = 5

        self.memory: list[dict] = []
        self.substitutions: dict[str, str] = {}
        self.char_maps: dict[str, dict[str, str]] = {}
        self.usage: dict[int, int] = {}

        self.load()

    # ---- Neural network core ----

    def _one_hot(self, c: str) -> np.ndarray:
        idx = self.char_to_idx.get(c, 0)
        v = np.zeros(self.vocab_size)
        v[idx] = 1.0
        return v

    def _ctx(self, text: str, pos: int) -> np.ndarray:
        half = self.context_size // 2
        vecs = []
        for i in range(pos - half, pos - half + self.context_size):
            if 0 <= i < len(text):
                vecs.append(self._one_hot(text[i]))
            else:
                vecs.append(np.zeros(self.vocab_size))
        return np.concatenate(vecs)

    def forward(self, X: np.ndarray) -> np.ndarray:
        h = np.maximum(0, X @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def _train_step(self, X: np.ndarray, y: np.ndarray, lr: float = 0.05):
        n = X.shape[0]
        h = np.maximum(0, X @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = e / np.sum(e, axis=1, keepdims=True)
        loss = -np.mean(np.log(probs[range(n), y] + 1e-8))

        dlogits = probs.copy()
        dlogits[range(n), y] -= 1
        dlogits /= n
        dW2 = h.T @ dlogits
        db2 = np.sum(dlogits, axis=0)
        dh = dlogits @ self.W2.T
        dh[h <= 0] = 0
        dW1 = X.T @ dh
        db1 = np.sum(dh, axis=0)

        for g in [dW1, db1, dW2, db2]:
            np.clip(g, -5.0, 5.0, out=g)

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        return loss

    def _build_char_examples(self):
        for wrong, correct in self.examples:
            m = min(len(wrong), len(correct))
            for p in range(m):
                if wrong[p] != correct[p]:
                    self.char_examples.append((self._ctx(wrong, p), self.char_to_idx.get(correct[p], 0)))
        self.examples.clear()

    def train(self, epochs: int = 10):
        self._build_char_examples()
        if not self.char_examples:
            self.pending_train = 0
            return
        X = np.array([e[0] for e in self.char_examples])
        y = np.array([e[1] for e in self.char_examples])
        log.info("NeuralBrain: training on %d char-examples...", len(self.char_examples))
        t0 = time.time()
        for ep in range(epochs):
            perm = np.random.permutation(len(X))
            Xs, ys = X[perm], y[perm]
            bs = min(64, len(X))
            losses = []
            for s in range(0, len(X), bs):
                losses.append(self._train_step(Xs[s:s+bs], ys[s:s+bs]))
            log.info("  Epoch %d/%d: loss=%.4f", ep+1, epochs, float(np.mean(losses)))
        self.total_trained += len(self.char_examples)
        self.pending_train = 0
        log.info("NeuralBrain: trained in %.2fs (total: %d)", time.time()-t0, self.total_trained)
        self.save()

    def _predict_char(self, text: str, pos: int) -> tuple[str | None, float]:
        if self.total_trained < 10:
            return None, 0.0
        probs = self.forward(self._ctx(text, pos).reshape(1, -1))[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        if conf < 0.4:
            return None, conf
        return self.idx_to_char[idx], conf

    # ---- Public interface (same as before) ----

    def think_text(self, raw: str) -> str:
        t = raw
        for wrong, right in self.substitutions.items():
            if wrong in t:
                t = t.replace(wrong, right)
        return t

    def think_fields(self, values: list[str], boxes: list[dict]) -> list[str]:
        from difflib import SequenceMatcher
        corrected = list(values)
        fuzzy_applied = neural_applied = 0

        for i, val in enumerate(values):
            if not val:
                continue
            field_id = boxes[i]["id"] if i < len(boxes) else f"field_{i}"

            # Phase 1 — neural character prediction
            if self.total_trained >= 10:
                out_chars, total_conf = [], 0.0
                for p in range(len(val)):
                    pred, conf = self._predict_char(val, p)
                    out_chars.append(pred if pred else val[p])
                    total_conf += conf
                avg_conf = total_conf / max(len(val), 1)
                neural_val = "".join(out_chars)
                if neural_val != val and avg_conf >= 0.5:
                    corrected[i] = neural_val
                    neural_applied += 1
                    continue

            # Phase 2 — exact match (fuzzy fallback)
            for mi, mem in enumerate(self.memory):
                if mem.get("field_id") != field_id:
                    continue
                if mem["wrong"] == val:
                    corrected[i] = mem["correct"]
                    self.usage[mi] = self.usage.get(mi, 0) + 1
                    fuzzy_applied += 1
                    break
            if corrected[i] != val:
                continue

            # Phase 3 — fuzzy match
            best_score, best_correct, best_mi = 0, "", -1
            for mi, mem in enumerate(self.memory):
                if mem.get("field_id") != field_id:
                    continue
                score = SequenceMatcher(None, val, mem["wrong"]).ratio()
                if score > best_score and score >= 0.7:
                    best_score, best_correct, best_mi = score, mem["correct"], mi
            if best_correct:
                corrected[i] = best_correct
                self.usage[best_mi] = self.usage.get(best_mi, 0) + 1
                fuzzy_applied += 1
                continue

            # Phase 4 — char map
            cmap = self.char_maps.get(field_id, {})
            if cmap:
                nv = "".join(cmap.get(c, c) for c in val)
                if nv != val:
                    corrected[i] = nv
                    fuzzy_applied += 1

        total = fuzzy_applied + neural_applied
        if total:
            log.info("NeuralBrain: corrected %d field(s) (neural=%d fuzzy=%d)", total, neural_applied, fuzzy_applied)
        return corrected

    def learn(self, raw_text: str, field_id: str, wrong_val: str, correct_val: str, source: str = "manual"):
        if not wrong_val or not correct_val or wrong_val == correct_val:
            return

        self.examples.append((wrong_val, correct_val))
        self.pending_train += 1

        self.memory.append({"raw": raw_text[:200], "field_id": field_id, "wrong": wrong_val, "correct": correct_val, "source": source})
        idx = len(self.memory) - 1
        self.usage[idx] = 0

        if len(wrong_val) >= 2:
            self.substitutions[wrong_val] = correct_val

        if field_id not in self.char_maps:
            self.char_maps[field_id] = {}
        for wc, cc in zip(wrong_val, correct_val):
            if wc != cc:
                old = self.char_maps[field_id].get(wc)
                if old is None or old == cc:
                    self.char_maps[field_id][wc] = cc

        log.info("NeuralBrain learned: '%s' -> '%s' (pending=%d)", wrong_val[:30], correct_val[:30], self.pending_train)

        if self.pending_train >= self.train_frequency:
            self.train(epochs=10)
        else:
            self.save()

    def learn_from_rag(self, raw_before, raw_after, parsed_before, parsed_after, boxes):
        if raw_before == raw_after:
            return
        for i, (old_val, new_val) in enumerate(zip(parsed_before, parsed_after)):
            if old_val != new_val and old_val and new_val:
                field_id = boxes[i]["id"] if i < len(boxes) else f"field_{i}"
                self.learn(raw_before, field_id, old_val, new_val, source="rag")
                break

    def load(self):
        if not _brain_path.exists():
            return
        try:
            data = json.loads(_brain_path.read_text(encoding="utf-8"))
            self.memory = data.get("memory", [])
            self.substitutions = data.get("substitutions", {})
            self.char_maps = data.get("char_maps", {})
            self.usage = {int(k): v for k, v in data.get("usage", {}).items()}

            w1 = data.get("nn_W1")
            if w1:
                self.W1 = np.array(w1)
                self.W2 = np.array(data["nn_W2"])
                self.b1 = np.array(data["nn_b1"])
                self.b2 = np.array(data["nn_b2"])
                self.total_trained = data.get("total_trained", 0)
                raw_ex = data.get("nn_examples", [])
                self.examples = [(a, b) for a, b in raw_ex]
                log.info("NeuralBrain loaded: %d memories, %d examples, %d trained",
                         len(self.memory), len(self.examples), self.total_trained)
            else:
                log.info("NeuralBrain loaded (cold): %d memories", len(self.memory))
        except Exception as e:
            log.warning("brain.json load failed: %s", e)

    def save(self):
        _brain_path.write_text(json.dumps({
            "memory": self.memory[-500:],
            "substitutions": self.substitutions,
            "char_maps": self.char_maps,
            "usage": self.usage,
            "nn_W1": self.W1.tolist(),
            "nn_W2": self.W2.tolist(),
            "nn_b1": self.b1.tolist(),
            "nn_b2": self.b2.tolist(),
            "nn_examples": self.examples[-500:],
            "total_trained": self.total_trained,
        }, indent=2), encoding="utf-8")



def merge_missing(dest: dict, src: dict):
    for k, v in src.items():
        if k not in dest:
            dest[k] = v
        elif isinstance(v, dict) and isinstance(dest.get(k), dict):
            merge_missing(dest[k], v)
        elif k == "rag_prompt":
            dest[k] = v


def validate_config(cfg: dict) -> list[str]:
    errors = []
    if not cfg.get("text_boxes"):
        errors.append("At least one text_box is required")
    region = cfg.get("ocr_image_region", {})
    for k in ("x", "y", "width", "height"):
        if not isinstance(region.get(k), (int, float)):
            errors.append(f"ocr_image_region.{k} must be a number")
    for i, box in enumerate(cfg.get("text_boxes", [])):
        for key in ("x", "y", "width", "height"):
            if not isinstance(box.get(key), (int, float)):
                errors.append(f"text_boxes[{i}].{key} must be a number")
    return errors


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        log.warning("Config not found at %s", CONFIG_PATH)
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
        log.info("Default config created at %s. Use --record to set up or edit coordinates.", CONFIG_PATH)
        sys.exit(0)
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    merge_missing(cfg, DEFAULT_CONFIG)
    errs = validate_config(cfg)
    for err in errs:
        log.error("Config error: %s", err)
    if errs:
        log.error("Fix config errors and re-run.")
        sys.exit(1)
    return cfg


def in_blacklist(x: int, y: int, buttons: list[dict]) -> bool:
    for b in buttons:
        if b["x"] <= x <= b["x"] + b["width"] and b["y"] <= y <= b["y"] + b["height"]:
            return True
    return False


def safe_move(x: int, y: int, buttons: list[dict], dur: float) -> bool:
    if in_blacklist(x, y, buttons):
        return False
    pyautogui.moveTo(x, y, duration=dur)
    return True


def capture_region_exact(x: int, y: int, w: int, h: int):
    arr = np.array(pyautogui.screenshot(region=(x, y, w, h)))
    return arr


def preprocess_image(img: np.ndarray) -> list[tuple[str, np.ndarray, str]]:
    """Returns list of (name, processed_image, tesseract_config) tuples."""
    variants = []
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    base_cfg = "--psm 6"

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", otsu, base_cfg))

    adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 2)
    variants.append(("adaptive_gaussian", adapt, base_cfg))

    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    _, den_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("denoised_otsu", den_otsu, base_cfg))

    bilat = cv2.bilateralFilter(gray, 9, 75, 75)
    _, bil_otsu = cv2.threshold(bilat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("bilateral_otsu", bil_otsu, base_cfg))

    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    _, sharp_otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("sharpened_otsu", sharp_otsu, base_cfg))

    _, bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(bin, kernel, iterations=1)
    variants.append(("dilated", dilated, base_cfg))

    # Contrast enhancement for better number differentiation
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, enhanced_bin = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("clahe_otsu", enhanced_bin, base_cfg))

    return variants


def ocr_with_confidence(img: np.ndarray, lang: str, config: str = "") -> tuple[str, float]:
    data = pytesseract.image_to_data(img, lang=lang, config=config, output_type=pytesseract.Output.DICT)
    confs = [c for c in data["conf"] if c > 0]
    avg_conf = float(np.mean(confs)) if confs else 0.0
    text = pytesseract.image_to_string(img, lang=lang, config=config)
    return text, avg_conf


def ocr_image(img: np.ndarray, lang: str, retries: int = 1, conf_threshold: float = 0) -> str:
    variants = preprocess_image(img)
    best_text, best_conf = "", 0.0

    for name, processed, config in variants:
        for attempt in range(1 + retries):
            text, conf = ocr_with_confidence(processed, lang, config)
            if not text.strip():
                continue
            if conf > best_conf:
                best_text, best_conf = text, conf
            if conf >= 95.0:
                return text
            if conf_threshold > 0 and conf >= conf_threshold:
                return text
            if attempt < retries:
                time.sleep(0.05)

    return best_text


def validate_line(line: str, pattern: str | None, field_label: str) -> bool:
    if not pattern:
        return True
    if re.fullmatch(pattern, line.strip()):
        return True
    log.warning("Validation failed for %s: '%s' does not match '%s'", field_label, line.strip()[:40], pattern)
    return False


US_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL","GA","HI","ID","IL",
    "IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE",
    "NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD",
    "TN","TX","UT","VT","VA","WA","WV","WI","WY",
}

def clean_ocr_text(text: str) -> str:
    """Remove non-printable symbols and normalize whitespace before parsing."""
    t = text
    # Remove trademark/registration/section symbols that Tesseract hallucinates
    t = t.replace("\u2122", "").replace("\u00AE", "").replace("\u00A7", " ")
    # Collapse multiple spaces
    t = re.sub(r" {2,}", " ", t).strip()
    return t


def parse_ocr_text(text: str) -> list[str]:
    """Parse OCR text (4 rows separated by newline or pipe) into 16 field values."""
    raw = clean_ocr_text(text.strip())
    if not raw:
        return []
    # Split by newline first; if only 1 row, try splitting by pipe
    rows = [r.strip() for r in raw.split("\n") if r.strip()]
    if len(rows) == 1 and "|" in raw:
        rows = [r.strip() for r in raw.split("|") if r.strip()]
    if len(rows) < 1:
        return []
    while len(rows) < 4:
        rows.append("")
    rows = rows[:4]
    log.info("OCR data rows: %d", len(rows))
    values = []

    # Row 1: First Name | Last Name | Email | SSN
    row = rows[0]
    email = ""
    at_pos = row.find("@")
    if at_pos >= 0:
        before_at = row[:at_pos].strip().split()[-1] if row[:at_pos].strip() else ""
        after_at = row[at_pos+1:].strip().split()[0] if row[at_pos+1:].strip() else ""
        if before_at and after_at:
            rest = row[at_pos+1:].strip()
            rest_parts = rest.split()
            if len(rest_parts) >= 2 and rest_parts[1].lower() in ("com", "net", "org", "edu", "gov", "in", "uk"):
                email = f"{before_at}@{rest_parts[0]}.{rest_parts[1]}"
            else:
                email = f"{before_at}@{after_at}"
    email_regex = re.search(r"\S+@\S+\.\S+", row)
    if email_regex and (not email or len(email_regex.group()) > len(email)):
        email = email_regex.group()
    if email:
        email_clean = re.sub(r"^[^a-zA-Z0-9]+", "", email)
        if email_clean:
            email = email_clean
        epos = row.find(email.split("@")[0]) if email.split("@")[0] in row else row.find("@")
        if epos < 0:
            epos = row.find("@") - len(email.split("@")[0]) if row.find("@") >= 0 else 0
        if epos < 0:
            epos = 0
        before = row[:epos].strip().rstrip("—").strip()
        name_parts = before.split()
        values.append(name_parts[0] if name_parts else "")
        values.append(name_parts[1] if len(name_parts) > 1 else "")
        values.append(email)
        # Find email end in original text more robustly
        after_raw = row[epos:].strip()
        # Remove email prefix + @ + domain from after_raw
        email_prefix = email.split("@")[0]
        after_raw = after_raw[len(email_prefix):].strip()
        if after_raw.startswith("@"):
            after_raw = after_raw[1:].strip()
        # Remove domain part(s)
        domain_parts = email.split("@")[1].split(".")
        for dp in domain_parts:
            if after_raw.lower().startswith(dp.lower()):
                after_raw = after_raw[len(dp):].strip()
        ssn_match = re.search(r"\d{6,}", after_raw)
        values.append(ssn_match.group() if ssn_match else "")
    else:
        parts = row.split()
        values.append(parts[0] if parts else "")
        values.append(parts[1] if len(parts) > 1 else "")
        values.append("")
        values.append("")

    # Row 2: Phone | Bank Name | A/C No | Loan amount
    row = rows[1].strip()
    nums = re.findall(r"\d{3,}", row)
    bank_text = row
    for n in nums:
        bank_text = bank_text.replace(n, "", 1).strip()
    bank_text = bank_text.strip("\" ").strip()
    values.append(nums[0] if len(nums) > 0 else "")
    values.append(bank_text if bank_text else "")
    values.append(nums[1] if len(nums) > 1 else "")
    values.append(nums[2] if len(nums) > 2 else "")

    # Row 3: Address | City | State | Zip
    row = rows[2].strip()
    if row.startswith("gapeachpas"):
        row = row[len("gapeachpas"):].strip()
    zip_match = re.search(r"\b(\d{5})\b", row)
    if zip_match:
        zip_val = zip_match.group(1)
        before_zip = row[:zip_match.start()].strip()
        state_val = ""
        words_before = before_zip.split()
        for w in reversed(words_before):
            wc = w.strip(".,;!?").upper()
            if len(wc) == 2 and wc.isalpha() and wc in US_STATES:
                state_val = wc
                break
        if state_val:
            ci_pos = before_zip.lower().rfind(state_val.lower())
            before_state = before_zip[:ci_pos].strip()
            city_parts = re.findall(r"\b([A-Z]{2,})\b", before_state)
            if city_parts:
                city_val = city_parts[-1]
                addr_end = before_state.rfind(city_val)
                addr = before_state[:addr_end].strip().rstrip(",").strip()
            else:
                city_val = ""
                addr = before_state
            values.append(addr)
            values.append(city_val)
            values.append(state_val)
            values.append(zip_val)
        else:
            values.extend([before_zip, "", "", zip_val])
    else:
        values.extend([row, "", "", ""])

    # Row 4: DOB | Licence No | Licence State | IP
    row = rows[3] if len(rows) > 3 else ""
    ip_match = re.search(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b", row)
    ip_val = ip_match.group(1) if ip_match else ""
    if not ip_val:
        ip_match2 = re.findall(r"(\d{1,3})\s*[.]?\s*(\d{1,3})\s+(\d{1,3})\s*[.]?\s*(\d{1,3})\b", row)
        for g in ip_match2:
            parts = [int(x) for x in g if x.isdigit()]
            if len(parts) == 4 and all(0 <= p <= 255 for p in parts):
                ip_val = f"{parts[0]}.{parts[1]}.{parts[2]}.{parts[3]}"
                break
    date_match = re.search(r"\b(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})\b", row)
    date_val = date_match.group(1) if date_match else ""
    all_state_matches = re.findall(r"\b([A-Za-z]{2})\b", row)
    state_val = ""
    for s in reversed(all_state_matches):
        su = s.upper()
        if su in US_STATES and (not ip_val or su != ip_val[:2]):
            state_val = su
            break
    licence_val = ""
    if date_val:
        after_date = row[row.index(date_val) + len(date_val):].strip()
        if ip_val and ip_val in after_date:
            after_date = after_date[:after_date.index(ip_val)].strip()
        licence_match = re.search(r"\b([A-Z]?\d{6,}[A-Z]?\d*)\b", after_date)
        if licence_match:
            licence_val = licence_match.group(1)
    values.append(date_val)
    values.append(licence_val)
    values.append(state_val)
    values.append(ip_val)

    return values


def _toggle_capslock():
    """Press Capslock key. Works on Windows."""
    import ctypes
    # Simulate Capslock key press via Win32 API
    ctypes.windll.user32.keybd_event(0x14, 0, 0, 0)   # key down
    ctypes.windll.user32.keybd_event(0x14, 0, 2, 0)   # key up


def is_capslock_on() -> bool:
    import ctypes
    return bool(ctypes.windll.user32.GetKeyState(0x14) & 1)


def type_text(text: str, cfg: dict):
    """Type text into active field. Handles Capslock automatically."""
    caps_was_on = is_capslock_on()
    if caps_was_on:
        _toggle_capslock()
        time.sleep(0.05)
    try:
        pyautogui.write(text, interval=cfg.get("typing_delay", 0.015))
    finally:
        if caps_was_on:
            time.sleep(0.05)
            _toggle_capslock()


def _fill_values(values: list[str], cfg: dict):
    global _cancel
    boxes, buttons = cfg["text_boxes"], cfg["blacklist_buttons"]
    filled = 0
    for i, box in enumerate(boxes):
        if _cancel:
            _cancel = False
            return
        if i >= len(values) or not values[i]:
            log.info("  Skip %s: empty", box["label"])
            continue
        cx = box["x"] + box["width"] // 2
        cy = box["y"] + box["height"] // 2
        if not safe_move(cx, cy, buttons, cfg["mouse_move_duration"]):
            log.warning("  Skip %s: blacklisted", box["label"])
            continue
        time.sleep(0.08)
        if cfg["click_before_type"]:
            pyautogui.click()
            time.sleep(0.05)
        type_text(str(values[i]), cfg)
        filled += 1
        log.info("  [%d] %s <- %s", filled, box["label"], str(values[i])[:40])
    log.info("Filled %d / %d fields", filled, len(boxes))
    if sys.platform == "win32":
        import winsound
        winsound.Beep(880, 100)


def fill_boxes(text: str, cfg: dict):
    global _cancel
    boxes, buttons = cfg["text_boxes"], cfg["blacklist_buttons"]
    lines = [ln for ln in text.split("\n") if ln.strip()]

    if not lines:
        log.warning("No text lines to type.")
        return

    n = min(len(lines), len(boxes))
    for i in range(n):
        if _cancel:
            _cancel = False
            return

        line = lines[i]
        box = boxes[i]
        pattern = box.get("validation")

        if pattern and not validate_line(line, pattern, box["label"]):
            continue

        cx = box["x"] + box["width"] // 2
        cy = box["y"] + box["height"] // 2
        if not safe_move(cx, cy, buttons, cfg["mouse_move_duration"]):
            continue
        time.sleep(0.08)
        if cfg["click_before_type"]:
            pyautogui.click()
            time.sleep(0.05)
        type_text(line, cfg)

    if sys.platform == "win32":
        import winsound
        winsound.Beep(880, 100)


def fill_from_json(data: dict, cfg: dict):
    global _cancel
    boxes, buttons = cfg["text_boxes"], cfg["blacklist_buttons"]
    filled = 0
    for box in boxes:
        if _cancel:
            _cancel = False
            return
        value = data.get(box["id"], "")
        if not value:
            log.info("  Skip %s: empty", box["label"])
            continue
        cx = box["x"] + box["width"] // 2
        cy = box["y"] + box["height"] // 2
        if not safe_move(cx, cy, buttons, cfg["mouse_move_duration"]):
            log.warning("  Skip %s: blacklisted", box["label"])
            continue
        time.sleep(0.08)
        if cfg["click_before_type"]:
            pyautogui.click()
            time.sleep(0.05)
        type_text(str(value), cfg)
        filled += 1
        log.info("  [%d] %s <- %s", filled, box["label"], str(value)[:40])

    log.info("Filled %d / %d fields", filled, len(boxes))
    if sys.platform == "win32":
        import winsound
        winsound.Beep(880, 100)


_brain = NeuralBrain()


def learn_correction():
    """Hotkey handler: user corrects fields manually, then presses learn shortcut."""
    global _last_filled_values, _last_ocr_text
    if not _last_filled_values or not _last_ocr_text:
        log.info("No previous fill data to learn from")
        return
    boxes = load_config().get("text_boxes", [])
    log.info("Learn mode: reading corrected fields...")
    learned = 0
    for i, box in enumerate(boxes):
        if i >= len(_last_filled_values):
            break
        old_val = _last_filled_values[i]
        if not old_val:
            continue
        # Click on field to read its value
        cx = box["x"] + box["width"] // 2
        cy = box["y"] + box["height"] // 2
        pyautogui.moveTo(cx, cy, duration=0.1)
        time.sleep(0.05)
        pyautogui.click()
        time.sleep(0.1)
        # Select all + copy
        pyautogui.hotkey("ctrl", "a")
        time.sleep(0.05)
        pyautogui.hotkey("ctrl", "c")
        time.sleep(0.1)
        # Read clipboard
        import subprocess
        try:
            proc = subprocess.Popen(["powershell", "-command", "Get-Clipboard"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            new_val = proc.communicate(timeout=3)[0].decode("utf-8", errors="replace").strip()
        except Exception:
            new_val = ""
        if new_val and new_val != old_val:
            _brain.learn(_last_ocr_text, box.get("id", f"field_{i}"), old_val, new_val, source="manual")
            learned += 1
    log.info("Learned %d field correction(s)", learned)


def warmup_ollama(cfg: dict) -> bool:
    """Ping ollama and pre-load model. Returns True if available."""
    api_url = cfg.get("rag_api_url", "http://localhost:11434/api/generate")
    model = cfg.get("rag_model", "llama3.2")
    base = api_url.rsplit("/api/", 1)[0]
    try:
        resp = requests.get(f"{base}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.RequestException:
        log.info("ollama not reachable (RAG unavailable)")
        return False
    # Pre-load model so first RAG call is fast
    try:
        t0 = time.time()
        resp = requests.post(
            api_url,
            json={"model": model, "prompt": ".", "stream": False},
            timeout=cfg.get("rag_timeout", 120),
        )
        resp.raise_for_status()
        log.info("ollama %s warmed up in %.1fs", model, time.time() - t0)
        return True
    except requests.RequestException as e:
        log.warning("ollama warmup failed: %s", e)
        return False


class RAGCorrector:
    def __init__(self, cfg: dict):
        self.api_url = cfg.get("rag_api_url", "http://localhost:11434/api/generate")
        self.model = cfg.get("rag_model", "llama3.2")
        self.retries = cfg.get("rag_retry_count", 1)
        self.timeout = cfg.get("rag_timeout", 120)
        self.prompt_template = cfg.get("rag_prompt", "Fix OCR errors: {text}")

    def correct(self, text: str) -> str:
        if not text.strip():
            return text

        prompt = self.prompt_template.replace("{text}", text)

        for attempt in range(1 + self.retries):
            try:
                t0 = time.time()
                resp = requests.post(
                    self.api_url,
                    json={"model": self.model, "prompt": prompt, "stream": False},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                elapsed = time.time() - t0
                corrected = resp.json().get("response", "").strip()
                if corrected:
                    log.info("RAG correction done in %.1fs (attempt %d)", elapsed, attempt + 1)
                    return corrected
            except requests.RequestException as e:
                log.warning("RAG attempt %d failed: %s", attempt + 1, e)
                if attempt < self.retries:
                    time.sleep(1)
        return text


def do_ocr_fill(cfg: dict, region: dict | None = None):
    global _busy, _cancel, _last_ocr_text, _last_filled_values
    if _busy:
        return
    _busy = True
    _cancel = False
    t0 = time.time()

    # Ping ollama on every trigger (fast) — model already warm from startup
    rag_available = False
    if cfg.get("enable_rag"):
        base = cfg["rag_api_url"].rsplit("/api/", 1)[0]
        try:
            resp = requests.get(f"{base}/api/tags", timeout=5)
            resp.raise_for_status()
            rag_available = True
        except requests.RequestException:
            log.info("ollama ping failed, skipping RAG")

    try:
        r = region or cfg.get("ocr_image_region", {})
        img = capture_region_exact(r["x"], r["y"], r["width"], r["height"])

        text = ocr_image(
            img,
            cfg["ocr_lang"],
            retries=cfg.get("ocr_retry_count", 1),
            conf_threshold=cfg.get("ocr_confidence_threshold", 0),
        )
        if text.strip():
            log.info("OCR (%d chars): %s", len(text), text[:200].replace("\n", " | "))
        else:
            log.warning("OCR result empty")

        # Step 0: Apply learned text-level corrections before parsing
        corrected_raw = _brain.think_text(text)

        # Step 1: Try parser directly
        parsed_values = parse_ocr_text(corrected_raw)
        # Step 1b: Apply learned field-level corrections
        parsed_values = _brain.think_fields(parsed_values, cfg["text_boxes"])
        non_empty_parsed = sum(1 for v in parsed_values if v)
        log.info("Parser: %d/%d fields extracted", non_empty_parsed, len(cfg["text_boxes"]))

        if non_empty_parsed >= len(cfg["text_boxes"]):
            _last_ocr_text = corrected_raw
            _last_filled_values = parsed_values[:len(cfg["text_boxes"])]
            _fill_values(_last_filled_values, cfg)
            log.info("Done in %.1fs", time.time() - t0)
            return

        # Step 2: Parser incomplete — try RAG to fix OCR errors, then re-parse
        if rag_available and corrected_raw.strip() and non_empty_parsed < len(cfg["text_boxes"]):
            log.info("Parser missed %d fields, trying RAG OCR fix...",
                     len(cfg["text_boxes"]) - non_empty_parsed)
            corrector = RAGCorrector(cfg)
            rag_result = corrector.correct(corrected_raw)
            if rag_result and rag_result != corrected_raw:
                log.info("RAG corrected OCR text, re-parsing...")
                re_parsed = parse_ocr_text(rag_result)
                re_parsed = _brain.think_fields(re_parsed, cfg["text_boxes"])
                re_non_empty = sum(1 for v in re_parsed if v)
                log.info("Re-parser: %d/%d fields", re_non_empty, len(cfg["text_boxes"]))
                if re_non_empty > non_empty_parsed:
                    # Brain learns from RAG improvement
                    _brain.learn_from_rag(corrected_raw, rag_result, parsed_values, re_parsed, cfg["text_boxes"])
                    parsed_values = re_parsed
                    non_empty_parsed = re_non_empty
            else:
                log.info("RAG returned same text or failed")

        if _cancel:
            _cancel = False
            return

        _last_ocr_text = corrected_raw
        _last_filled_values = parsed_values
        if non_empty_parsed > 0:
            _fill_values(parsed_values, cfg)
        else:
            fill_boxes(corrected_raw, cfg)
        log.info("Done in %.1fs", time.time() - t0)
    except Exception as e:
        log.exception("Error: %s", e)
    finally:
        _busy = False


def cancel_op():
    global _cancel
    _cancel = True


def on_click(cfg: dict, x: int, y: int, button, pressed):
    if not pressed or button != mouse.Button.left or _busy:
        return

    region = cfg.get("ocr_image_region", {})
    rx, ry, rw, rh = region.get("x", 0), region.get("y", 0), region.get("width", 400), region.get("height", 200)

    if not (rx <= x <= rx + rw and ry <= y <= ry + rh):
        return

    ct = cfg.get("click_to_trigger", True)
    keys = [k.strip() for k in cfg.get("start_shortcut", "space+ctrl").split("+")]
    modifiers_held = all(keyboard.is_pressed(k) for k in keys)

    if ct or modifiers_held:
        log.info("Triggered: click on image region (%d, %d)", x, y)
        do_ocr_fill(cfg)


def start_mouse_listener(cfg: dict):
    global _mouse_listener
    _mouse_listener = mouse.Listener(on_click=lambda x, y, button, pressed: on_click(cfg, x, y, button, pressed))
    _mouse_listener.daemon = True
    _mouse_listener.start()
    return _mouse_listener


def record_setup():
    print()
    print("=" * 46)
    print("     INTERACTIVE SETUP (RECORD MODE)")
    print("=" * 46)
    print()
    print("  [1]      → Mark corner of the OCR image (top-left, then bottom-right)")
    print("  [2]      → Mark TEXT FIELD (place cursor inside each text box)")
    print("              Order matters: 1st OCR line -> 1st field, 2nd line -> 2nd field")
    print("  [3]      → Mark BLACKLISTED BUTTON (buttons to never click)")
    print("  [4]      → Done adding items, go to next step")
    print("  [0]      → SAVE to config.json & exit")
    print("  [9]      → Exit without saving")
    print()
    print("  Trigger: hold space+ctrl + click on image -> instant OCR+fill")
    print()

    cfg_data = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.exists() else json.dumps(DEFAULT_CONFIG, indent=2)
    config = json.loads(cfg_data)
    config["text_boxes"] = []
    config["blacklist_buttons"] = []
    config["click_to_trigger"] = True
    config["start_shortcut"] = "space+ctrl"
    config["ocr_lang"] = input("OCR language (default: eng): ").strip() or "eng"

    field_count = [0]
    btn_count = [0]

    print("\nReady! Move your mouse and press keys.\n")

    stop = threading.Event()

    def _show():
        while not stop.is_set():
            x, y = pyautogui.position()
            print(f"\r  Mouse: X={x:>4} Y={y:>4}   Fields: {field_count[0]}   Buttons: {btn_count[0]}   ", end="", flush=True)
            time.sleep(0.08)

    t = threading.Thread(target=_show, daemon=True)
    t.start()

    # --- OCR Image Region ---
    print("  >> Place cursor at TOP-LEFT corner of the image, press 1")
    keyboard.wait("1")
    x1, y1 = pyautogui.position()
    print(f"\n  Top-left: ({x1}, {y1})")

    print("  >> Move cursor to BOTTOM-RIGHT corner of the image, press 1 again")
    keyboard.wait("1")
    x2, y2 = pyautogui.position()
    print(f"  Bottom-right: ({x2}, {y2})")

    ox = min(x1, x2)
    oy = min(y1, y2)
    ow = abs(x2 - x1)
    oh = abs(y2 - y1)
    if ow < 20 or oh < 20:
        print("  [!] Region too small, using default 400x200 around click point")
        ox, oy = x1 - 200, y1 - 100
        ow, oh = 400, 200
    config["ocr_image_region"] = {"x": ox, "y": oy, "width": ow, "height": oh}
    print(f"  ✓ OCR image region: ({ox}, {oy}) {ow}x{oh}")

    # --- Text Fields ---
    print("\n  >> Move cursor to each TEXT FIELD, press 2")
    print("  >> Press 4 when done adding fields.")
    stop.set()  # stop live cursor display, input() will conflict otherwise
    keyboard.wait("2")
    while True:
        fx, fy = pyautogui.position()
        field_count[0] += 1
        name = input(f"  Field {field_count[0]} name (e.g., 'First Name', 'Last Name'): ").strip() or f"Field {field_count[0]}"
        w = input(f"  {name} width (default: 300): ").strip() or "300"
        h = input(f"  {name} height (default: 30): ").strip() or "30"
        fid = name.lower().replace(" ", "_")
        config["text_boxes"].append({
            "id": fid,
            "x": fx, "y": fy,
            "width": int(w), "height": int(h),
            "label": name,
            "validation": None,
        })
        print(f"  ✓ {name} at ({fx}, {fy})")
        print("  >> Press 2 for next field, or 4 to stop.")
        time.sleep(0.3)  # clear residual key events from input()

        done = threading.Event()
        result = [None]

        def _wait():
            while True:
                k = keyboard.read_event()
                if k.event_type == "down":
                    result[0] = k.name
                    done.set()
                    return

        kw = threading.Thread(target=_wait, daemon=True)
        kw.start()
        done.wait()
        if result[0] == "4":
            break

    # --- Blacklist Buttons ---
    print("\n  >> Now mark BUTTONS to avoid. Move cursor over each, press 3.")
    print("  >> Press 4 when done.")
    keyboard.wait("3")

    while True:
        bx, by = pyautogui.position()
        btn_count[0] += 1
        bw = input(f"  Button {btn_count[0]} width (default: 100): ").strip() or "100"
        bh = input(f"  Button {btn_count[0]} height (default: 40): ").strip() or "40"
        config["blacklist_buttons"].append({
            "id": f"btn_{btn_count[0]}", "x": bx, "y": by,
            "width": int(bw), "height": int(bh),
        })
        print(f"  ✓ Button {btn_count[0]} at ({bx}, {by})")
        print("  >> Press 3 for next button, or 4 to stop.")
        time.sleep(0.3)

        done = threading.Event()
        result = [None]

        def _wait2():
            while True:
                k = keyboard.read_event()
                if k.event_type == "down":
                    result[0] = k.name
                    done.set()
                    return

        kw2 = threading.Thread(target=_wait2, daemon=True)
        kw2.start()
        done.wait()
        if result[0] == "4":
            break

    stop.set()

    print()
    print("-" * 30)
    print("  Config Preview")
    print("-" * 30)
    print(json.dumps(config, indent=2))
    print()
    print("  Press 0  -> SAVE to config.json & exit")
    print("  Press 9  -> Cancel (no save)")
    print("  Press Esc -> Cancel")
    print()

    # Wait for save or cancel
    while True:
        k = keyboard.read_event()
        if k.event_type != "down":
            continue
        if k.name == "0":
            import sys
            CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
            print(f"\n  ✓ Saved to {CONFIG_PATH}")
            print("  Run: python ocr_automation.py")
            sys.exit(0)
        elif k.name in ("9", "esc"):
            print("\n  Cancelled. No changes saved.")
            sys.exit(0)


def run_debug():
    cfg = load_config()
    region = cfg.get("ocr_image_region", {})
    rx, ry, rw, rh = region.get("x", 0), region.get("y", 0), region.get("width", 400), region.get("height", 200)
    print()
    print("=" * 46)
    print("  DEBUG MODE - Click anywhere to test")
    print("=" * 46)
    print(f"  Image region: ({rx}, {ry}) to ({rx+rw}, {ry+rh})")
    print(f"  Fields:")
    for b in cfg.get("text_boxes", []):
        print(f"    {b['label']}: ({b['x']}, {b['y']}) {b['width']}x{b['height']}")
    print()
    print("  Click inside image region -> would trigger OCR+fill")
    print("  Click outside -> shows coordinates")
    print("  Press ESC to quit")
    print()

    def on_debug_click(x, y, button, pressed):
        if not pressed:
            return
        inside = "INSIDE" if (rx <= x <= rx + rw and ry <= y <= ry + rh) else "OUTSIDE"
        sp = keyboard.is_pressed("space")
        ct = keyboard.is_pressed("ctrl")
        print(f"  Click at ({x:>4}, {y:>4})  {inside} region  space={sp}  ctrl={ct}")

    listener = mouse.Listener(on_click=on_debug_click)
    listener.daemon = True
    listener.start()
    keyboard.wait("esc")
    listener.stop()


def calibrate():
    print()
    print("-" * 30)
    print("  CALIBRATION MODE")
    print("-" * 30)
    print()
    print("Move mouse over your text boxes and buttons.")
    print("Note down the coordinates, then edit config.json.")
    print("Press Q to quit.\n")
    stop = threading.Event()

    def _show():
        while not stop.is_set():
            x, y = pyautogui.position()
            print(f"\rX: {x:>4}  Y: {y:>4}   ", end="", flush=True)
            time.sleep(0.08)

    t = threading.Thread(target=_show, daemon=True)
    t.start()
    keyboard.wait("q")
    stop.set()
    print("\n\nCalibration ended.")
    print("Edit config.json with the coordinates you noted.")


def main():
    ap = argparse.ArgumentParser(description="OCR Desktop Automation Tool")
    ap.add_argument("--calibrate", "-c", action="store_true",
                    help="Show live mouse coordinates (press Q to quit)")
    ap.add_argument("--record", "-r", action="store_true",
                    help="Interactive setup -- press keys to record positions")
    ap.add_argument("--debug-clicks", "-d", action="store_true",
                    help="Show click coordinates and region status (for testing)")
    args = ap.parse_args()

    print()
    print("=" * 46)
    print("  OCR Desktop Automation Tool")
    print("  Click image -> auto-fills text fields")
    print("=" * 46)
    print()

    log.info("Logging to %s", LOG_PATH)

    if args.record:
        record_setup()
        return

    if args.calibrate:
        calibrate()
        return

    if args.debug_clicks:
        run_debug()
        return

    cfg = load_config()

    if cfg.get("tesseract_path"):
        tp = Path(cfg["tesseract_path"])
        if tp.exists():
            pytesseract.pytesseract.tesseract_cmd = str(tp)

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05

    # Auto-detect ollama at startup — enables RAG if available
    log.info("Checking ollama...")
    rag_ok = warmup_ollama(cfg)
    if rag_ok:
        cfg["enable_rag"] = True
        log.info("ollama ready — RAG auto-fix active")
    else:
        cfg["enable_rag"] = False
        log.info("ollama not available — RAG auto-fix disabled")

    region = cfg.get("ocr_image_region", {})
    ct = cfg.get("click_to_trigger", True)
    log.info("Image region: (%d, %d) %dx%d", region.get("x", 0), region.get("y", 0), region.get("width", 0), region.get("height", 0))
    log.info("%d text box(es), %d blacklisted button(s)",
             len(cfg["text_boxes"]), len(cfg["blacklist_buttons"]))

    xk, ck = cfg["stop_shortcut"], cfg.get("cancel_shortcut", "esc")
    sk = cfg.get("start_shortcut", "space+ctrl")
    log.info("Hold %s + click on image -> OCR & fill fields", sk)
    log.info("Hotkey: %s -> exit | %s -> cancel", xk, ck)

    start_mouse_listener(cfg)
    keyboard.add_hotkey(ck, cancel_op)
    lk = cfg.get("learn_shortcut", "ctrl+shift+c")
    keyboard.add_hotkey(lk, learn_correction)
    log.info("Hotkey: %s -> learn correction", lk)
    keyboard.wait(xk)

    if _mouse_listener:
        _mouse_listener.stop()
    while _busy:
        time.sleep(0.05)
    log.info("Exited.")


if __name__ == "__main__":
    main()

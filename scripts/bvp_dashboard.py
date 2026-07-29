from __future__ import annotations

import json
import math
import pickle
import threading
import webbrowser
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "features" / "3.5_peak_corrected"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed" / "all_preprocessed" / "3.5"
PREDICTION_CSV = PROJECT_ROOT / "outputs" / "3.5_peak_corrected" / "predictions.csv"

HOST = "127.0.0.1"
PORT = 8765

BVP_HZ = 64
MAX_POINTS_PER_SUBJECT = 9000
VISIBLE_SECONDS = 240.0
ANIMATION_SECONDS_PER_SUBJECT = 160.0

LABEL_NAMES = {
    0: "stress",
    1: "non-stress",
    2: "ignore",
}

LABEL_COLORS = {
    0: "#e5484d",
    1: "#30a46c",
    2: "#8b8f97",
}


HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>BVP Graph with Stress</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #1f2937;
      --muted: #6b7280;
      --line: #d9dee7;
      --accent: #2563eb;
      --stress: #e5484d;
      --nonstress: #30a46c;
      --ignore: #8b8f97;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .shell {
      width: min(1440px, calc(100vw - 28px));
      margin: 0 auto;
      padding: 12px 0;
    }

    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 10px;
    }

    h1 {
      margin: 0;
      font-size: 21px;
      font-weight: 700;
      letter-spacing: 0;
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 286px;
      gap: 12px;
      align-items: stretch;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
    }

    .chart-panel {
      min-height: 500px;
      padding: 12px;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 10px;
    }

    .chart-toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      min-height: 32px;
    }

    #chartTitle {
      font-size: 14px;
      font-weight: 650;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .controls {
      display: flex;
      align-items: center;
      gap: 8px;
      flex: 0 0 auto;
    }

    button {
      height: 32px;
      min-width: 68px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font-weight: 600;
      cursor: pointer;
    }

    button:hover { border-color: #b8c0cf; }

    canvas {
      width: 100%;
      height: 100%;
      min-height: 430px;
      max-height: calc(100vh - 104px);
      display: block;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfe;
    }

    .side {
      padding: 14px;
      display: grid;
      grid-template-rows: auto auto auto minmax(0, 1fr);
      gap: 12px;
      min-height: 500px;
    }

    .legend {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fafafa;
    }

    .legend-half + .legend-half {
      border-left: 1px solid var(--line);
      padding-left: 12px;
    }

    .label {
      margin: 0 0 10px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }

    .legend-row {
      display: flex;
      align-items: center;
      gap: 9px;
      min-height: 26px;
      font-size: 14px;
      font-weight: 700;
    }

    .stress-score {
      min-height: 78px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 4px;
    }

    #stressScoreValue {
      font-size: 30px;
      line-height: 1;
      font-weight: 800;
      color: var(--stress);
    }

    #stressScoreState {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }

    .swatch {
      width: 12px;
      height: 12px;
      border-radius: 999px;
      flex: 0 0 auto;
    }

    .swatch.stress { background: var(--stress); }
    .swatch.nonstress { background: var(--nonstress); }
    .swatch.ignore { background: var(--ignore); }

    .kv {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px 14px;
      padding-top: 2px;
      font-size: 13px;
    }

    .kv span:nth-child(odd) { color: var(--muted); }
    .kv span:nth-child(even) { font-weight: 650; }

    .progress {
      height: 10px;
      border-radius: 999px;
      background: #e8ebf0;
      overflow: hidden;
    }

    #progressBar {
      width: 0%;
      height: 100%;
      background: var(--accent);
      transition: width 0.08s linear;
    }

    .subject-list {
      min-height: 0;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fbfcfe;
    }

    .subject-row {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 7px 8px;
      border-radius: 5px;
      color: var(--muted);
      font-size: 13px;
      cursor: pointer;
    }

    .subject-row.active {
      color: var(--ink);
      background: #eef4ff;
      font-weight: 700;
    }

    @media (max-width: 900px) {
      .layout { grid-template-columns: 1fr; }
      header { align-items: start; flex-direction: column; }
      .side { min-height: auto; }
      canvas { min-height: 380px; max-height: none; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div>
        <h1>BVP Graph with Stress</h1>
      </div>
    </header>

    <section class="layout">
      <div class="panel chart-panel">
        <div class="chart-toolbar">
          <div id="chartTitle">Loading...</div>
          <div class="controls">
            <button id="prevBtn">Prev</button>
            <button id="pauseBtn">Pause</button>
            <button id="nextBtn">Next</button>
          </div>
        </div>
        <canvas id="chart"></canvas>
      </div>

      <aside class="panel side">
        <div class="legend">
          <div class="legend-half">
            <p class="label">Label Colors</p>
            <div class="legend-row"><span class="swatch stress"></span><span>stress</span></div>
            <div class="legend-row"><span class="swatch nonstress"></span><span>non-stress</span></div>
            <div class="legend-row"><span class="swatch ignore"></span><span>ignore</span></div>
          </div>
          <div class="legend-half">
            <p class="label">Stress Score</p>
            <div class="stress-score">
              <div id="stressScoreValue">-</div>
              <div id="stressScoreState">stress 구간에서 표시</div>
            </div>
          </div>
        </div>

        <div class="kv">
          <span>Subject</span><span id="subjectValue">-</span>
          <span>Position</span><span id="positionValue">-</span>
          <span>Duration</span><span id="durationValue">-</span>
          <span>BVP Hz</span><span id="hzValue">-</span>
          <span>Visible Time</span><span id="visibleValue">-</span>
        </div>

        <div class="progress"><div id="progressBar"></div></div>

        <div class="subject-list" id="subjectList"></div>
      </aside>
    </section>
  </main>

  <script>
    const config = {
      animationSeconds: Number("{{ANIMATION_SECONDS}}"),
      visibleSeconds: Number("{{VISIBLE_SECONDS}}")
    };

    const chart = document.getElementById("chart");
    const ctx = chart.getContext("2d");
    const prevBtn = document.getElementById("prevBtn");
    const pauseBtn = document.getElementById("pauseBtn");
    const nextBtn = document.getElementById("nextBtn");

    let dataset = null;
    let current = null;
    let subjectIndex = 0;
    let startedAt = 0;
    let paused = false;
    let pauseOffset = 0;

    function resizeCanvas() {
      const rect = chart.getBoundingClientRect();
      const scale = window.devicePixelRatio || 1;
      chart.width = Math.max(1, Math.floor(rect.width * scale));
      chart.height = Math.max(1, Math.floor(rect.height * scale));
      ctx.setTransform(scale, 0, 0, scale, 0, 0);
      draw();
    }

    function labelColor(label) {
      if (label === 0) return "#e5484d";
      if (label === 1) return "#30a46c";
      return "#8b8f97";
    }

    function setText(id, value) {
      document.getElementById(id).textContent = value;
    }

    function activeLabelAt(timeSec) {
      if (!current || current.times.length === 0) return null;
      let bestIndex = 0;
      let bestDistance = Infinity;
      for (let i = 0; i < current.times.length; i++) {
        const distance = Math.abs(current.times[i] - timeSec);
        if (distance < bestDistance) {
          bestDistance = distance;
          bestIndex = i;
        }
      }
      return current.labels[bestIndex];
    }

    function stressWindowAt(timeSec) {
      if (!current || !current.stress_windows) return null;
      let bestWindow = null;
      let bestDistance = Infinity;
      for (const item of current.stress_windows) {
        if (timeSec >= item.start_seconds && timeSec <= item.end_seconds) {
          const center = (item.start_seconds + item.end_seconds) / 2;
          const distance = Math.abs(timeSec - center);
          if (distance < bestDistance) {
            bestDistance = distance;
            bestWindow = item;
          }
        }
      }
      return bestWindow;
    }

    function currentProgress(now = performance.now()) {
      if (!current) return 0;
      return Math.min(1, Math.max(0, (now - startedAt) / 1000 / config.animationSeconds));
    }

    function visibleRange(progress) {
      const visibleSeconds = Math.min(config.visibleSeconds, current.duration_seconds);
      const travel = Math.max(0, current.duration_seconds - visibleSeconds);
      const tStart = travel * progress;
      const tEnd = tStart + visibleSeconds;
      return { tStart, tEnd, visibleSeconds };
    }

    function updateSidebar(progress = 0) {
      if (!current || !dataset) return;
      const { tStart, tEnd, visibleSeconds } = visibleRange(progress);
      const centerTime = (tStart + tEnd) / 2;
      const centerLabel = activeLabelAt(centerTime);
      const stressWindow = centerLabel === 0 ? stressWindowAt(centerTime) : null;
      setText("subjectValue", current.subject);
      setText("positionValue", `${subjectIndex + 1} / ${dataset.subjects.length}`);
      setText("durationValue", `${current.duration_seconds.toFixed(1)}s`);
      setText("hzValue", `${current.bvp_hz} Hz`);
      setText("visibleValue", `${visibleSeconds.toFixed(0)}s`);
      setText("chartTitle", `${current.subject} | ${tStart.toFixed(1)}s - ${tEnd.toFixed(1)}s`);
      setText("stressScoreValue", stressWindow ? `${stressWindow.stress_score_0_100.toFixed(1)}` : "-");
      setText(
        "stressScoreState",
        stressWindow
          ? `${stressWindow.start_seconds.toFixed(0)}-${stressWindow.end_seconds.toFixed(0)}s`
          : "stress 구간에서 표시"
      );
      document.getElementById("progressBar").style.width = `${progress * 100}%`;

      document.querySelectorAll(".subject-row").forEach(row => {
        row.classList.toggle("active", Number(row.dataset.index) === subjectIndex);
      });
    }

    function drawGrid(width, height, pad) {
      ctx.strokeStyle = "#e2e6ee";
      ctx.lineWidth = 1;
      for (let i = 0; i <= 5; i++) {
        const y = pad.top + ((height - pad.top - pad.bottom) * i / 5);
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(width - pad.right, y);
        ctx.stroke();
      }
      for (let i = 0; i <= 8; i++) {
        const x = pad.left + ((width - pad.left - pad.right) * i / 8);
        ctx.beginPath();
        ctx.moveTo(x, pad.top);
        ctx.lineTo(x, height - pad.bottom);
        ctx.stroke();
      }
    }

    function drawAxes(width, height, pad, yMin, yMax, tStart, tEnd) {
      ctx.fillStyle = "#374151";
      ctx.font = "12px Segoe UI, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Time (seconds)", pad.left + (width - pad.left - pad.right) / 2, height - 10);

      ctx.save();
      ctx.translate(16, pad.top + (height - pad.top - pad.bottom) / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("BVP value", 0, 0);
      ctx.restore();

      ctx.textAlign = "right";
      for (let i = 0; i <= 5; i++) {
        const y = pad.top + ((height - pad.top - pad.bottom) * i / 5);
        const value = yMax - ((yMax - yMin) * i / 5);
        ctx.fillText(value.toFixed(2), pad.left - 8, y + 4);
      }

      ctx.textAlign = "center";
      for (let i = 0; i <= 8; i++) {
        const x = pad.left + ((width - pad.left - pad.right) * i / 8);
        const value = tStart + ((tEnd - tStart) * i / 8);
        ctx.fillText(value.toFixed(0), x, height - pad.bottom + 17);
      }
    }

    function drawLabelBands(tStart, tEnd, width, height, pad) {
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;
      let runLabel = null;
      let runStart = null;
      for (let i = 0; i < current.times.length; i++) {
        const t = current.times[i];
        if (t < tStart || t > tEnd) continue;
        const label = current.labels[i];
        if (runLabel === null) {
          runLabel = label;
          runStart = t;
        } else if (label !== runLabel) {
          fillBand(runStart, t, runLabel, tStart, tEnd, plotW, plotH, pad);
          runLabel = label;
          runStart = t;
        }
      }
      if (runLabel !== null) {
        fillBand(runStart, tEnd, runLabel, tStart, tEnd, plotW, plotH, pad);
      }
    }

    function fillBand(start, end, label, tStart, tEnd, plotW, plotH, pad) {
      const x = pad.left + ((start - tStart) / (tEnd - tStart)) * plotW;
      const w = Math.max(1, ((end - start) / (tEnd - tStart)) * plotW);
      ctx.fillStyle = labelColor(label) + "18";
      ctx.fillRect(x, pad.top, w, plotH);
    }

    function drawSignal(tStart, tEnd, width, height, pad, yMin, yMax) {
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;
      let previousLabel = null;
      let hasPoint = false;

      for (let i = 0; i < current.samples.length; i++) {
        const t = current.times[i];
        if (t < tStart || t > tEnd) continue;
        const label = current.labels[i];
        const x = pad.left + ((t - tStart) / (tEnd - tStart)) * plotW;
        const y = pad.top + (1 - ((current.samples[i] - yMin) / (yMax - yMin))) * plotH;

        if (!hasPoint || label !== previousLabel) {
          if (hasPoint) ctx.stroke();
          ctx.beginPath();
          ctx.strokeStyle = labelColor(label);
          ctx.lineWidth = 1.8;
          ctx.moveTo(x, y);
          hasPoint = true;
          previousLabel = label;
        } else {
          ctx.lineTo(x, y);
        }
      }
      if (hasPoint) ctx.stroke();
    }

    function draw(progress = 0) {
      const rect = chart.getBoundingClientRect();
      const width = rect.width;
      const height = rect.height;
      ctx.clearRect(0, 0, width, height);

      if (!current || !current.samples || current.samples.length < 2) {
        ctx.fillStyle = "#6b7280";
        ctx.font = "15px Segoe UI, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Loading subject BVP...", width / 2, height / 2);
        return;
      }

      const pad = { left: 72, right: 20, top: 20, bottom: 46 };
      const { tStart, tEnd } = visibleRange(progress);
      drawLabelBands(tStart, tEnd, width, height, pad);
      drawGrid(width, height, pad);
      drawAxes(width, height, pad, current.y_min, current.y_max, tStart, tEnd);
      drawSignal(tStart, tEnd, width, height, pad, current.y_min, current.y_max);
    }

    async function loadDataset() {
      const response = await fetch("/api/dataset");
      dataset = await response.json();

      const list = document.getElementById("subjectList");
      list.innerHTML = "";
      dataset.subjects.forEach((subject, index) => {
        const row = document.createElement("div");
        row.className = "subject-row";
        row.dataset.index = index;
        row.innerHTML = `<span>${subject.subject}</span><span>${subject.duration_seconds.toFixed(0)}s</span>`;
        row.addEventListener("click", () => loadSubject(index));
        list.appendChild(row);
      });
    }

    async function loadSubject(index) {
      subjectIndex = (index + dataset.subjects.length) % dataset.subjects.length;
      const response = await fetch(`/api/subject?index=${subjectIndex}`);
      current = await response.json();
      startedAt = performance.now();
      pauseOffset = 0;
      updateSidebar(0);
      draw(0);
    }

    async function nextSubject() {
      if (!dataset) return;
      await loadSubject(subjectIndex + 1);
    }

    async function prevSubject() {
      if (!dataset) return;
      await loadSubject(subjectIndex - 1);
    }

    function animate(now) {
      if (!paused && current && dataset) {
        const progress = currentProgress(now);
        updateSidebar(progress);
        draw(progress);
        if (progress >= 1) nextSubject();
      }
      requestAnimationFrame(animate);
    }

    pauseBtn.addEventListener("click", () => {
      paused = !paused;
      pauseBtn.textContent = paused ? "Play" : "Pause";
      if (paused) {
        pauseOffset = performance.now() - startedAt;
      } else {
        startedAt = performance.now() - pauseOffset;
      }
    });

    nextBtn.addEventListener("click", nextSubject);
    prevBtn.addEventListener("click", prevSubject);
    window.addEventListener("resize", resizeCanvas);

    (async function init() {
      await loadDataset();
      resizeCanvas();
      await loadSubject(0);
      requestAnimationFrame(animate);
    })();
  </script>
</body>
</html>
"""


class SubjectBvpData:
    def __init__(self, feature_dir: Path, preprocessed_dir: Path) -> None:
        self.feature_dir = feature_dir
        self.preprocessed_dir = preprocessed_dir
        self.predictions = load_prediction_scores(PREDICTION_CSV)
        self.subjects = self._load_subjects()
        if not self.subjects:
            raise RuntimeError("no matching subjects found")

    def _load_subjects(self) -> list[str]:
        subjects = []
        for feature_path in sorted(self.feature_dir.glob("S*.npz"), key=subject_sort_key):
            subject = feature_path.stem
            pkl_path = self.preprocessed_dir / subject / f"{subject}.pkl"
            if pkl_path.exists():
                subjects.append(subject)
        return subjects

    def dataset_payload(self) -> dict:
        subjects = []
        for subject in self.subjects:
            bvp_len = self._subject_bvp_length(subject)
            subjects.append(
                {
                    "subject": subject,
                    "duration_seconds": bvp_len / BVP_HZ,
                }
            )
        return {
            "subjects": subjects,
            "visible_seconds": VISIBLE_SECONDS,
            "animation_seconds_per_subject": ANIMATION_SECONDS_PER_SUBJECT,
        }

    def subject_payload(self, index: int) -> dict:
        subject_index = index % len(self.subjects)
        subject = self.subjects[subject_index]
        pkl_path = self.preprocessed_dir / subject / f"{subject}.pkl"
        with pkl_path.open("rb") as f:
            data = pickle.load(f)

        bvp = np.asarray(data["signal"]["wrist"]["BVP"], dtype=np.float32).reshape(-1)
        labels = np.asarray(data["label_bvp"], dtype=np.int8).reshape(-1)
        length = min(len(bvp), len(labels))
        bvp = bvp[:length]
        labels = labels[:length]

        bvp, labels = downsample_signal_and_labels(bvp, labels, MAX_POINTS_PER_SUBJECT)
        times = np.arange(len(bvp), dtype=np.float32) * (length / BVP_HZ / max(1, len(bvp)))
        y_min = float(np.min(bvp))
        y_max = float(np.max(bvp))
        if y_min == y_max:
            y_min -= 0.5
            y_max += 0.5
        margin = max(0.01, (y_max - y_min) * 0.08)

        return {
            "subject_index": subject_index,
            "subject": subject,
            "bvp_hz": BVP_HZ,
            "duration_seconds": length / BVP_HZ,
            "y_min": y_min - margin,
            "y_max": y_max + margin,
            "samples": bvp.round(6).tolist(),
            "labels": labels.astype(int).tolist(),
            "times": times.round(3).tolist(),
            "stress_windows": self._subject_stress_windows(subject),
        }

    def _subject_bvp_length(self, subject: str) -> int:
        pkl_path = self.preprocessed_dir / subject / f"{subject}.pkl"
        with pkl_path.open("rb") as f:
            data = pickle.load(f)
        return int(np.asarray(data["signal"]["wrist"]["BVP"]).reshape(-1).shape[0])

    def _subject_stress_windows(self, subject: str) -> list[dict]:
        feature_path = self.feature_dir / f"{subject}.npz"
        prediction_rows = self.predictions.get(subject, {})
        if not feature_path.exists() or not prediction_rows:
            return []

        data = np.load(feature_path, allow_pickle=True)
        starts = np.asarray(data["start_seconds"], dtype=np.float64)
        ends = np.asarray(data["end_seconds"], dtype=np.float64)
        labels = np.asarray(data["y"], dtype=np.int8)

        windows = []
        for sample_index, row in prediction_rows.items():
            if sample_index < 0 or sample_index >= len(starts):
                continue
            if int(labels[sample_index]) != 0:
                continue
            windows.append(
                {
                    "sample_index": sample_index,
                    "start_seconds": float(starts[sample_index]),
                    "end_seconds": float(ends[sample_index]),
                    "stress_score": float(row["stress_score"]),
                    "stress_score_0_100": float(row["stress_score_0_100"]),
                    "y_pred": int(row["y_pred"]),
                    "correct": bool(row["correct"]),
                }
            )
        return windows


def load_prediction_scores(path: Path) -> dict[str, dict[int, dict]]:
    if not path.exists():
        return {}

    predictions: dict[str, dict[int, dict]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject = row["test_subject"]
            sample_index = int(row["sample_index"])
            stress_score = float(row["stress_score"])
            score_0_100_text = row.get("stress_score_0_100")
            score_0_100 = (
                float(score_0_100_text)
                if score_0_100_text not in (None, "")
                else decision_score_to_0_100(stress_score)
            )
            predictions.setdefault(subject, {})[sample_index] = {
                "y_pred": int(row["y_pred"]),
                "stress_score": stress_score,
                "stress_score_0_100": score_0_100,
                "correct": int(row.get("correct", "0")) == 1,
            }
    return predictions


def decision_score_to_0_100(score: float) -> float:
    return 100.0 / (1.0 + math.exp(-score))


def subject_sort_key(value: str | Path) -> tuple[int, str]:
    name = value.stem if isinstance(value, Path) else str(value)
    suffix = name[1:] if name.startswith("S") else name
    return (int(suffix) if suffix.isdigit() else math.inf, name)


def downsample_signal_and_labels(
    samples: np.ndarray,
    labels: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(samples) <= max_points:
        return samples, labels
    step = int(math.ceil(len(samples) / max_points))
    usable = (len(samples) // step) * step
    sample_bins = samples[:usable].reshape(-1, step)
    label_bins = labels[:usable].reshape(-1, step)
    downsampled_samples = sample_bins.mean(axis=1)
    downsampled_labels = np.apply_along_axis(majority_label, 1, label_bins).astype(np.int8)
    return downsampled_samples, downsampled_labels


def majority_label(values: np.ndarray) -> int:
    labels, counts = np.unique(values, return_counts=True)
    return int(labels[np.argmax(counts)])


def make_handler(data: SubjectBvpData) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                html = (
                    HTML.replace("{{ANIMATION_SECONDS}}", str(ANIMATION_SECONDS_PER_SUBJECT))
                    .replace("{{VISIBLE_SECONDS}}", str(VISIBLE_SECONDS))
                )
                self.write_html(html)
                return

            if parsed.path == "/api/dataset":
                self.write_json(data.dataset_payload())
                return

            if parsed.path == "/api/subject":
                query = parse_qs(parsed.query)
                index = int(query.get("index", ["0"])[0])
                self.write_json(data.subject_payload(index))
                return

            self.send_error(404, "not found")

        def log_message(self, format: str, *args) -> None:
            return

        def write_html(self, text: str) -> None:
            body = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def write_json(self, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return DashboardHandler


def main() -> None:
    data = SubjectBvpData(FEATURE_DIR, PREPROCESSED_DIR)
    server = ThreadingHTTPServer((HOST, PORT), make_handler(data))
    url = f"http://{HOST}:{PORT}"

    print(f"BVP dashboard: {url}")
    print(f"features:     {FEATURE_DIR}")
    print(f"preprocessed: {PREPROCESSED_DIR}")
    print("Press Ctrl+C to stop.")

    threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    server.serve_forever()


if __name__ == "__main__":
    main()

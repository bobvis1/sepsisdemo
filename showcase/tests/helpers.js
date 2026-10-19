// Helpers for the showcase browser tests: launch the installed Chrome, open the built page with
// URL hooks, collect page errors, and recompute expected values independently of the page.
import puppeteer from "puppeteer-core";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
export const PAGE = path.join(ROOT, "outputs/showcase/standalone.html");
export const DATA_PATH = path.join(ROOT, "outputs/showcase/data.json");
export const SCREENS = path.join(ROOT, "outputs/showcase/screens");
const MAC_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

if (!fs.existsSync(PAGE) || !fs.existsSync(DATA_PATH)) {
  throw new Error(`Built page not found. Run the Python build first (see showcase/tests/README.md): ${PAGE}`);
}
export const D = JSON.parse(fs.readFileSync(DATA_PATH, "utf8"));

export function chromePath() {
  const p = process.env.CHROME_PATH || MAC_CHROME;
  if (!fs.existsSync(p)) {
    throw new Error(`Chrome not found at ${p}. Install Google Chrome or set CHROME_PATH to its executable.`);
  }
  return p;
}
export const launch = () => puppeteer.launch({ executablePath: chromePath(), headless: true });

export const tid = (id) => `[data-testid="${id}"]`;

/** Open the built page with URL hooks and wait for body[data-ready="1"]. Errors are collected on page.errors. */
export async function open(browser, query = "still=1", { width = 1440, height = 900, scheme = "light", reducedMotion = false } = {}) {
  const page = await browser.newPage();
  page.errors = [];
  page.on("pageerror", (e) => page.errors.push("pageerror: " + e.message));
  page.on("console", (m) => { if (m.type() === "error") page.errors.push("console.error: " + m.text()); });
  await page.setViewport({ width, height, deviceScaleFactor: 1 });
  await page.emulateMediaFeatures([
    { name: "prefers-color-scheme", value: scheme },
    { name: "prefers-reduced-motion", value: reducedMotion ? "reduce" : "no-preference" },
  ]);
  await page.goto(`file://${PAGE}${query ? "?" + query : ""}`);
  await page.waitForSelector('body[data-ready="1"]', { timeout: 20000 });
  return page;
}
export const text = (page, id) => page.$eval(tid(id), (el) => el.textContent.trim());
export const state = (page) => page.evaluate(() => JSON.parse(JSON.stringify(window.__showcase)));
export const attr = (page, id, name) => page.$eval(tid(id), (el, n) => el.getAttribute(n), name);

// ---- expected values, computed here from data.json (formatting mirrors what a reader sees) -----
export const fmtInt = (n) => Math.round(n).toLocaleString("en-US");
export const pct = (v) => (100 * v).toFixed(1) + "%";
export const riskPct = (t) => (100 * t).toPrecision(3) + "%";
export const COHORT = { B: "B", A: "A_val" };
export const A_INDEX = D.thresholds.indexOf(D.meta.threshold);
export const BUDGET_INDEX = D.thresholds.indexOf(D.meta.budget_threshold);

export function expectedCounters(cohort, ti) {
  const c = D.cohorts[COHORT[cohort]], g = D.grid[COHORT[cohort]][ti], ns = c.n_septic, nn = c.n - ns;
  const of = (k, d) => `${fmtInt(g[k])} of ${fmtInt(d)} (${pct(g[k] / d)})`;
  return {
    "count-early": of("early", ns), "count-late": of("late", ns), "count-missed": of("missed", ns),
    "count-false_alarm": of("false_alarm", nn), "count-quiet": of("quiet", nn),
    "metric-threshold": riskPct(D.thresholds[ti]),
    "metric-fa": g.false_alarms_per_patient_day.toFixed(2),
    "metric-utility": g.utility.toFixed(3),
    "metric-window": pct(g.septic_alerted_in_window),
  };
}
export async function readCounters(page) {
  const out = {};
  for (const id of Object.keys(expectedCounters("B", A_INDEX))) out[id] = await text(page, id);
  return out;
}

// ---- an independent implementation of the night-shift simulation (same rules as the page) -------
function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
export function shiftTally(seed, thr) {
  const pool = D.shift_pool, rnd = mulberry32(seed), alert = (p, i) => p.risk[i] != null && p.risk[i] >= thr;
  const t = { episodes: 0, episodesSeptic: 0, onsets: 0, onsetsAlerted: 0, alertHours: 0 };
  for (let b = 0; b < 20; b++) {
    let pid = Math.floor(rnd() * pool.length), idx = Math.floor(rnd() * pool[pid].risk.length);
    for (let h = 0; h < 24; h++) {
      const p = pool[pid];
      if (alert(p, idx)) { t.alertHours++; if (!(idx > 0 && alert(p, idx - 1))) { t.episodes++; if (p.septic) t.episodesSeptic++; } }
      if (p.septic && p.onset_idx === idx) { t.onsets++; for (let j = Math.max(0, idx - 48); j < idx; j++) if (alert(p, j)) { t.onsetsAlerted++; break; } }
      idx++;
      if (idx >= p.risk.length) { pid = Math.floor(rnd() * pool.length); idx = 0; }
    }
  }
  return t;
}

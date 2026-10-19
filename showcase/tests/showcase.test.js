import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import {
  D, A_INDEX, BUDGET_INDEX, SCREENS, launch, open, text, state, attr, tid,
  expectedCounters, readCounters, riskPct, fmtInt, pct, shiftTally,
} from "./helpers.js";

let browser;
before(async () => { browser = await launch(); });
after(async () => { await browser?.close(); });
const using = async (query, opts, fn) => { const page = await open(browser, query, opts); try { return await fn(page); } finally { await page.close(); } };
const handleCenter = async (page) => {
  await page.evaluate(() => document.getElementById("field").scrollIntoView({ block: "center" })); // real mouse events need it on screen
  const b = await (await page.$(tid("threshold-handle"))).boundingBox(); return { x: b.x + b.width / 2, y: b.y + b.height / 2 }; };
const dragTo = async (page, dx) => {
  const c = await handleCenter(page);
  await page.mouse.move(c.x, c.y); await page.mouse.down();
  await page.mouse.move(c.x + dx / 2, c.y, { steps: 5 }); await page.mouse.move(c.x + dx, c.y, { steps: 5 }); await page.mouse.up();
};
const nums = (c) => ({ early: +c["count-early"].split(" ")[0].replace(/,/g, ""), fa: +c["count-false_alarm"].split(" ")[0].replace(/,/g, "") });

test("1 loads clean at desktop width, with the intro animation too", async () => {
  await using("still=1", {}, async (p) => assert.deepEqual(p.errors, []));
  await using("speed=20", {}, async (p) => { assert.deepEqual(p.errors, []); assert.equal((await state(p)).ready, true); });
});

test("2 default numbers match the evaluation and the drawn dots match the counters", async () => {
  await using("still=1", {}, async (p) => {
    assert.deepEqual(await readCounters(p), expectedCounters("B", A_INDEX));
    const ev = D.meta.evaluation.B;
    assert.equal(await text(p, "hero-patients"), fmtInt(ev.n_patients));
    assert.equal(await text(p, "hero-caught"), pct(ev.septic_alerted_before_onset));
    assert.equal(await text(p, "hero-fa"), ev.false_alarms_per_patient_day.toFixed(2));
    const g = D.grid.B[A_INDEX], s = await state(p);
    for (const k of ["early", "late", "missed", "false_alarm", "quiet"]) assert.equal(s.counts[k], g[k], k);
    assert.equal(g.early / D.cohorts.B.n_septic, ev.septic_alerted_before_onset);
  });
});

test("3 dragging the threshold with the mouse moves it and every counter follows the grid", async () => {
  await using("still=1", {}, async (p) => {
    const before = nums(await readCounters(p)), v0 = +(await attr(p, "threshold-handle", "aria-valuenow"));
    await dragTo(p, 160);
    const s = await state(p), v1 = +(await attr(p, "threshold-handle", "aria-valuenow")), after = nums(await readCounters(p));
    assert.ok(v1 > v0 && s.thresholdIndex > A_INDEX);
    assert.ok(after.early <= before.early && after.fa <= before.fa);
    assert.deepEqual(await readCounters(p), expectedCounters("B", s.thresholdIndex));
    await dragTo(p, -2000); // far past the left edge
    const s2 = await state(p), left = nums(await readCounters(p));
    assert.equal(s2.thresholdIndex, 0);
    assert.ok(left.early >= after.early && left.fa >= after.fa);
    assert.deepEqual(await readCounters(p), expectedCounters("B", 0));
    assert.deepEqual(p.errors, []);
  });
});

test("4 keyboard: three ArrowRight presses move exactly three grid points", async () => {
  await using("still=1", {}, async (p) => {
    await p.focus(tid("threshold-handle"));
    for (let i = 0; i < 3; i++) await p.keyboard.press("ArrowRight");
    assert.equal((await state(p)).thresholdIndex, A_INDEX + 3);
    assert.deepEqual(await readCounters(p), expectedCounters("B", A_INDEX + 3));
    assert.equal(+(await attr(p, "threshold-handle", "aria-valuenow")), D.thresholds[A_INDEX + 3]);
  });
});

test("5 reset returns to the hospital A threshold; the operating-point buttons snap to theirs", async () => {
  await using("still=1", {}, async (p) => {
    await dragTo(p, 120); await p.click(tid("reset"));
    assert.equal((await state(p)).thresholdIndex, A_INDEX);
    assert.deepEqual(await readCounters(p), expectedCounters("B", A_INDEX));
    await p.click(tid("op-budget"));
    assert.equal((await state(p)).thresholdIndex, BUDGET_INDEX);
    assert.deepEqual(await readCounters(p), expectedCounters("B", BUDGET_INDEX));
  });
});

test("6 cohort toggle shows hospital A (358 septic) and restores hospital B", async () => {
  await using("still=1", {}, async (p) => {
    await p.click(tid("cohort-A"));
    assert.equal((await state(p)).cohort, "A_val");
    assert.deepEqual(await readCounters(p), expectedCounters("A", A_INDEX));
    assert.match(await text(p, "count-early"), /of 358 /);
    await p.click(tid("cohort-B"));
    assert.deepEqual(await readCounters(p), expectedCounters("B", A_INDEX));
    assert.deepEqual(p.errors, []);
  });
});

test("7 URL hooks open in the same state as clicking there", async () => {
  const t = D.thresholds[BUDGET_INDEX];
  const viaHook = await using(`still=1&cohort=A&t=${t}`, {}, (p) => readCounters(p));
  const viaClicks = await using("still=1", {}, async (p) => { await p.click(tid("cohort-A")); await p.click(tid("op-budget")); return readCounters(p); });
  assert.deepEqual(viaHook, viaClicks);
  assert.deepEqual(viaHook, expectedCounters("A", BUDGET_INDEX));
});

test("8 replay: every patient ends on the banner for its category, with its drivers", async () => {
  await using("speed=50", {}, async (p) => {
    await p.evaluate(() => document.getElementById("scene-replay").scrollIntoView());
    for (let i = 0; i < D.replay.length; i++) {
      await p.waitForSelector('#rp-monitor[data-done="1"]', { timeout: 15000 });
      const pt = D.replay[i], banner = await text(p, "replay-banner");
      assert.equal(await text(p, "replay-patient"), pt.id);
      if (pt.category === "early") assert.ok(banner.startsWith(`${Math.round(pt.lead_h)} hours of warning`), banner);
      if (pt.category === "late") assert.match(banner, /^Alert came (at the hour of onset|\d+ hours after onset)/);
      if (pt.category === "missed") assert.ok(banner.startsWith("No alert before onset"));
      if (pt.category === "false_alarm") assert.ok(banner.startsWith("No sepsis: false alarm"));
      if (pt.category === "quiet") assert.ok(banner.startsWith("No alert, no sepsis"));
      assert.equal(await p.$eval(tid("replay-alert-tag"), (e) => !e.hidden), pt.first_alert != null);
      const names = await p.$$eval('#rp-drivers .drv span', (els) => els.map((e) => e.textContent));
      assert.deepEqual(names, pt.drivers.map((d) => d.label));
      assert.ok(names.length >= 1);
      if (i < D.replay.length - 1) await p.click(tid("replay-next"));
    }
    assert.deepEqual(p.errors, []);
  });
});

test("9 replay: Pause stops the replay, Play resumes it (slow real mouse click)", async () => {
  await using("", {}, async (p) => {
    await p.evaluate(() => document.getElementById("scene-replay").scrollIntoView({ block: "center" }));
    await p.waitForFunction(() => window.__showcase.replay.playing && window.__showcase.replay.iculos > 2, { timeout: 10000 });
    const b = await (await p.$("#rp-play")).boundingBox(), x = b.x + b.width / 2, y = b.y + b.height / 2;
    const press = async () => { await p.mouse.move(x, y); await p.mouse.down(); await new Promise((r) => setTimeout(r, 120)); await p.mouse.up(); };
    await press();
    const s1 = await state(p);
    assert.equal(s1.replay.playing, false);
    assert.equal(await p.$eval("#rp-play", (e) => e.textContent), "Play");
    await new Promise((r) => setTimeout(r, 600));
    assert.equal((await state(p)).replay.iculos, s1.replay.iculos, "the hour must not advance while paused");
    await press();
    await p.waitForFunction((h) => window.__showcase.replay.iculos > h, { timeout: 5000 }, s1.replay.iculos);
    assert.equal(await p.$eval("#rp-play", (e) => e.textContent), "Pause");
  });
});

test("10 replay controls are not rewritten every frame (that can swallow clicks)", async () => {
  await using("", {}, async (p) => {
    await p.evaluate(() => document.getElementById("scene-replay").scrollIntoView({ block: "center" }));
    const counts = await p.evaluate(async () => {
      const ids = ["rp-play", "rp-banner", "rp-alert", "rp-id", "rp-cat"], n = {};
      const obs = ids.map((id) => { n[id] = 0; const o = new MutationObserver((m) => { n[id] += m.length; }); o.observe(document.getElementById(id), { childList: true, subtree: true, characterData: true, attributes: true }); return o; });
      document.getElementById("rp-replay").click();
      await new Promise((r) => setTimeout(r, 1200));
      obs.forEach((o) => o.disconnect());
      return n;
    });
    for (const [id, n] of Object.entries(counts)) assert.ok(n <= 2, `${id} changed ${n} times in 1.2 s of playback`);
  });
});

test("11 night shift: reproducible, matches an independent simulation, honest about the budget threshold", async () => {
  const thrA = D.meta.threshold, thrB = D.meta.budget_threshold, sum = { a: 0, b: 0, ahA: 0, ahB: 0 };
  for (const seed of [1, 2, 3, 4, 5, 6]) {
    const one = await using(`still=1&seed=${seed}`, {}, async (p) => {
      const s = await state(p), dom = {
        episodes: +(await text(p, "shift-episodes")), onsets: +(await text(p, "shift-onsets")), alertHours: +(await text(p, "shift-alert-hours")),
      };
      assert.equal(dom.episodes, s.shift.tally.episodes);
      assert.deepEqual(s.shift.tally, shiftTally(seed, thrA), `seed ${seed} at the A threshold`);
      await p.click(tid("shift-thr-budget"));
      assert.deepEqual((await state(p)).shift.tally, shiftTally(seed, thrB), `seed ${seed} at the budget threshold`);
      return [s.shift.tally, (await state(p)).shift.tally];
    });
    const again = await using(`still=1&seed=${seed}`, {}, async (p) => (await state(p)).shift.tally);
    assert.deepEqual(again, one[0], "same seed, same shift");
    assert.ok(one[1].alertHours <= one[0].alertHours, "a higher threshold never adds alert hours");
    sum.a += one[0].episodes; sum.b += one[1].episodes; sum.ahA += one[0].alertHours; sum.ahB += one[1].alertHours;
  }
  assert.ok(sum.b <= sum.a, `across seeds the budget threshold starts no more episodes (${sum.b} vs ${sum.a})`);
});

test("12 reduced motion skips the intro; dark mode changes the tokens, not the data", async () => {
  await using("", { reducedMotion: true }, async (p) => {
    assert.equal(await text(p, "hero-patients"), fmtInt(D.meta.evaluation.B.n_patients));
    assert.deepEqual(await readCounters(p), expectedCounters("B", A_INDEX));
  });
  const bg = {}, early = {};
  for (const scheme of ["light", "dark"]) {
    await using("still=1", { scheme }, async (p) => {
      bg[scheme] = await p.evaluate(() => getComputedStyle(document.body).backgroundColor);
      early[scheme] = await p.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue("--c-early").trim());
      assert.deepEqual(await readCounters(p), expectedCounters("B", A_INDEX));
      assert.deepEqual(p.errors, []);
    });
  }
  assert.notEqual(bg.light, bg.dark);
  assert.equal(early.light, "#1f5fbf"); assert.equal(early.dark, "#3a78dd");
  await using("still=1&theme=dark", { scheme: "light" }, async (p) =>
    assert.equal(await p.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue("--c-early").trim()), "#3a78dd"));
});

test("13 swimmer wall shows every septic patient and the never-alerted count", async () => {
  await using("still=1", {}, async (p) => {
    const s = await state(p);
    assert.equal(s.wall.rows, D.cohorts.B.n_septic);
    assert.equal(s.wall.neverAlerted, D.grid.B[A_INDEX].missed);
    assert.match(await text(p, "wall-count"), new RegExp(`${fmtInt(s.wall.rows)} septic patients · ${fmtInt(s.wall.neverAlerted)} never alerted`));
  });
});

test("14 screenshots for review (not assertions)", async () => {
  fs.mkdirSync(SCREENS, { recursive: true });
  for (const scheme of ["light", "dark"]) {
    await using("still=1&scene2=early&seed=3", { scheme }, async (p) => {
      for (const [name, sel] of [["hero", ".hero"], ["ward", "#scene-ward"], ["replay", "#scene-replay"], ["shift", "#scene-shift"], ["wall", "#scene-wall"], ["footer", "footer"]]) {
        await (await p.$(sel)).screenshot({ path: `${SCREENS}/${name}-${scheme}.png` });
      }
    });
  }
});

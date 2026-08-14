/* 信号时间轴：small-multiples 车道（不同量纲不共享 y 轴），
   缺陷窗口红色高亮、计划转场虚线标注，crosshair + tooltip，点击跳转视频。 */
const LANES = [
  ['luminance', '亮度 luminance'],
  ['diff_d1', '帧差 diff_d1'],
  ['flicker', '闪烁 flicker'],
  ['clip_dist', 'CLIP 距离'],
  ['warp_residual', '光流 warp 残差(全帧均值)'],
  ['warp_block_max', '局部块残差 max(T11)'],
];
const LANE_H = 58, GAP = 14, PAD_L = 10, PAD_R = 10, AXIS_H = 22;

function drawTimeline(container, ev, video) {
  const prev = ev.signals_preview;
  const dur = ev.scan_meta.duration_s;
  const series = prev.series;
  const nPts = series.luminance.length;
  const tOf = i => i * prev.stride / prev.fps;

  const lanes = LANES.filter(([k]) => (series[k] || []).length);
  const canvas = document.createElement('canvas');
  const tip = document.createElement('div');
  tip.className = 'tl-tip';
  container.classList.add('timeline');
  container.appendChild(canvas);
  container.appendChild(tip);

  const H = AXIS_H + lanes.length * (LANE_H + GAP);
  let W = 0;

  function xOf(t, w) { return PAD_L + t / dur * (w - PAD_L - PAD_R); }

  function render() {
    W = container.clientWidth;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr; canvas.height = H * dpr;
    canvas.style.height = H + 'px';
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    const css = getComputedStyle(document.documentElement);
    const C = k => css.getPropertyValue(k).trim();

    ctx.clearRect(0, 0, W, H);
    // 顶部时间轴
    ctx.strokeStyle = C('--baseline'); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PAD_L, AXIS_H - 6); ctx.lineTo(W - PAD_R, AXIS_H - 6); ctx.stroke();
    ctx.fillStyle = C('--muted'); ctx.font = '11px system-ui';
    const step = dur > 12 ? 2 : 1;
    for (let t = 0; t <= dur; t += step) {
      const x = xOf(t, W);
      ctx.fillRect(x - 0.5, AXIS_H - 9, 1, 3);
      ctx.fillText(t + 's', x - 6, AXIS_H - 12);
    }
    lanes.forEach(([key, label], li) => {
      const y0 = AXIS_H + li * (LANE_H + GAP);
      const data = series[key] || [];
      const mx = Math.max(...data, 1e-6), mn = Math.min(...data, 0);
      // 车道底
      ctx.fillStyle = C('--surface');
      ctx.fillRect(PAD_L, y0, W - PAD_L - PAD_R, LANE_H);
      // 转场窗（先画在底上）
      (ev.transitions || []).forEach(tr => {
        const x1 = xOf(Math.max(0, tr.start_s), W), x2 = xOf(Math.min(dur, tr.end_s || tr.start_s), W);
        ctx.fillStyle = 'rgba(137,135,129,0.18)';
        ctx.fillRect(x1, y0, Math.max(x2 - x1, 2), LANE_H);
      });
      // 缺陷窗
      (ev.findings || []).forEach(f => {
        const x1 = xOf(Math.max(0, f.start_s), W), x2 = xOf(Math.min(dur, f.end_s), W);
        ctx.fillStyle = 'rgba(208,59,59,0.22)';
        ctx.fillRect(x1, y0, Math.max(x2 - x1, 3), LANE_H);
      });
      // 曲线
      ctx.strokeStyle = C('--s1'); ctx.lineWidth = 1.6; ctx.beginPath();
      data.forEach((v, i) => {
        const x = xOf(tOf(i), W);
        const y = y0 + LANE_H - 4 - (v - mn) / (mx - mn + 1e-9) * (LANE_H - 12);
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.stroke();
      // 车道标签 + 量程
      ctx.fillStyle = C('--ink-2'); ctx.font = '11.5px system-ui';
      ctx.fillText(label, PAD_L + 5, y0 + 12);
      ctx.fillStyle = C('--muted');
      ctx.fillText(`max ${mx.toFixed(mx < 1 ? 3 : 1)}`, W - PAD_R - 70, y0 + 12);
    });
    // 转场虚线（跨车道）
    (ev.transitions || []).forEach(tr => {
      const x = xOf(tr.start_s, W);
      ctx.setLineDash([4, 4]); ctx.strokeStyle = C('--muted'); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, AXIS_H); ctx.lineTo(x, H - GAP); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = C('--muted'); ctx.font = '10.5px system-ui';
      ctx.fillText(tr.type, x + 3, AXIS_H + 10);
    });
  }

  let raf = null;
  canvas.addEventListener('pointermove', e => {
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const t = Math.max(0, Math.min(dur, (px - PAD_L) / (W - PAD_L - PAD_R) * dur));
    const idx = Math.max(0, Math.min(nPts - 1, Math.round(t * prev.fps / prev.stride)));
    if (raf) cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => {
      render();
      const ctx = canvas.getContext('2d');
      const x = xOf(tOf(idx), W);
      ctx.strokeStyle = 'rgba(255,255,255,0.55)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, AXIS_H); ctx.lineTo(x, H - GAP); ctx.stroke();
      tip.style.display = 'block';
      tip.style.left = Math.min(px + 14, W - 170) + 'px';
      tip.style.top = (e.clientY - rect.top + 10) + 'px';
      tip.textContent = '';
      const tt = document.createElement('div');
      tt.appendChild(Object.assign(document.createElement('b'), { textContent: tOf(idx).toFixed(2) + 's' }));
      tip.appendChild(tt);
      lanes.forEach(([key, label]) => {
        const row = document.createElement('div');
        const k = document.createElement('span'); k.className = 'k';
        k.textContent = label.split(' ')[0] + ' ';
        const v = document.createElement('b');
        const val = (series[key] || [])[idx];
        v.textContent = val === undefined ? '—' : val.toFixed(val < 1 ? 3 : 1);
        row.appendChild(k); row.appendChild(v); tip.appendChild(row);
      });
    });
  });
  canvas.addEventListener('pointerleave', () => { tip.style.display = 'none'; render(); });
  canvas.addEventListener('click', e => {
    if (!video) return;
    const rect = canvas.getBoundingClientRect();
    const t = (e.clientX - rect.left - PAD_L) / (W - PAD_L - PAD_R) * dur;
    video.currentTime = Math.max(0, Math.min(dur, t));
    video.play().catch(() => {});
  });
  new ResizeObserver(render).observe(container);
  render();

  const leg = document.createElement('div');
  leg.className = 'legend';
  [['sw', 'var(--s1)', '信号曲线'], ['bx', 'rgba(208,59,59,0.5)', '缺陷窗口'],
   ['bx', 'rgba(137,135,129,0.35)', '计划转场（豁免 T4）']].forEach(([cls, color, label]) => {
    const s = document.createElement('span');
    const sw = document.createElement('span'); sw.className = cls; sw.style.background = color;
    s.appendChild(sw); s.appendChild(document.createTextNode(label));
    leg.appendChild(s);
  });
  container.after(leg);
}

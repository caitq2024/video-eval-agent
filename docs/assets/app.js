/* 共享：数据加载、徽章、findings 表、计时条 */
const TYPE_ZH = {
  T0_file_gate: 'T0 文件损坏', T1_jump: 'T1 跳帧', T2_flicker: 'T2 闪烁',
  T3_freeze: 'T3 冻结', T4_unexpected_cut: 'T4 意外切换', T5_out_of_frame: 'T5 主体出界',
  T6_black: 'T6 黑帧', T7_deform: 'T7 主体变形', T8_identity_drift: 'T8 身份漂移',
  T9_vanish: 'T9 凭空消失', T10_misalignment: 'T10 语义不符',
  T11_local_incoherence: 'T11 局部时序不连贯', T17_motion_dynamics: 'T17 运动动态性',
  T19_cross_shot: 'T19 跨镜头一致性', T20_pipeline: 'T20 管线执行',
  vlm_defect: 'VLM 判定缺陷',
};
const STAGE_ZH = {
  t0_gate_s: 'T0 文件gate', s1_scan_s: 'S1 全帧扫描', s2_detect_s: 'S2 直判',
  s3_probes_s: 'S3 主体探针', s4_fuse_s: 'S4 信号融合',
  s4_vlm_s: 'S4/S5 VLM并行裁决(含T10)', s5_t10_s: 'S5 语义对齐',
};
const STAGE_COLOR = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#9085e9', '#e66767'];

async function loadData() {
  const r = await fetch('data/data.json');
  return r.json();
}
function q(k) { return new URLSearchParams(location.search).get(k); }
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function scoreBadge(score) {
  const b = el('span', 'badge');
  if (score === null || score === undefined) { b.classList.add('b-na'); b.textContent = '未评估'; return b; }
  const lv = score >= 85 ? ['b-good', '良好'] : score >= 70 ? ['b-warning', '注意']
    : score >= 50 ? ['b-serious', '较差'] : ['b-critical', '严重'];
  b.classList.add(lv[0]);
  const d = el('span', 'dot'); b.appendChild(d);
  b.appendChild(document.createTextNode(`${score} ${lv[1]}`));
  return b;
}
function fmtS(x) { return x === undefined || x === null ? '—' : (Math.round(x * 10) / 10) + 's'; }

function findingsTable(findings, onSeek) {
  const t = el('table');
  const th = el('tr');
  ['类型', '时间窗', '严重度', '证据', '裁决方'].forEach((h, i) => {
    const c = el('th', i === 2 ? 'num' : '', h); th.appendChild(c);
  });
  t.appendChild(th);
  if (!findings.length) {
    const tr = el('tr'); const td = el('td', '', '未检出缺陷 ✓');
    td.colSpan = 5; td.style.color = 'var(--good)'; tr.appendChild(td); t.appendChild(tr);
    return t;
  }
  findings.forEach(f => {
    const tr = el('tr', 'clickable');
    tr.appendChild(el('td', '', TYPE_ZH[f.type] || f.type));
    tr.appendChild(el('td', 'num', `${f.start_s}–${f.end_s}s`));
    const sv = el('td', 'num'); sv.appendChild(el('span', `sev sev${f.severity}`, 'S' + f.severity));
    tr.appendChild(sv);
    tr.appendChild(el('td', '', f.evidence || ''));
    const vb = el('td');
    const vbLabel = { detector: '直判', vlm: 'VLM', dual: '双重验证', llm: 'LLM文本' }[f.verdict_by] || f.verdict_by;
    vb.appendChild(el('span', `tag ${f.verdict_by}`, vbLabel));
    tr.appendChild(vb);
    tr.onclick = () => onSeek && onSeek(f.start_s);
    t.appendChild(tr);
  });
  return t;
}

function timingBar(timing, container) {
  const stages = Object.keys(STAGE_ZH).filter(k => timing[k] > 0);
  const total = stages.reduce((a, k) => a + timing[k], 0);
  const bar = el('div', 'tbar');
  const leg = el('div', 'tlegend');
  stages.forEach((k, i) => {
    const d = el('div');
    d.style.width = (timing[k] / total * 100) + '%';
    d.style.background = STAGE_COLOR[i % STAGE_COLOR.length];
    d.title = `${STAGE_ZH[k]} ${fmtS(timing[k])}`;
    bar.appendChild(d);
    const li = el('span');
    const sw = el('span', 'bx'); sw.style.background = STAGE_COLOR[i % STAGE_COLOR.length];
    sw.classList.add('bx'); li.appendChild(sw);
    li.appendChild(document.createTextNode(`${STAGE_ZH[k]} ${fmtS(timing[k])}`));
    leg.appendChild(li);
  });
  container.appendChild(bar); container.appendChild(leg);
}

function videoSrc(pid, model, name) { return `assets/videos/${pid}_${model}_${name}`; }

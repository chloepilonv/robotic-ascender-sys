// app/bms_ui/bms.js — BATTERY MANAGEMENT SYSTEM panel for app/web/index.html.
// Self-contained: injects its own CSS, panel and sidebar knobs, and hooks the
// page by wrapping paintHud(). Remove the <script> tag and nothing else changes.
// Data comes from app/bms_ui/bridge.py via the websocket `state` message (live)
// and from hud.json `bms_*` arrays (replay).
// Displays exactly: electrical power, battery life (SOC), pack temperature,
// and a foldable "torque and velocity of joints". ⓘ = math or source (MATH.md).
(function () {
  const css = `
    #bmsPanel { grid-row: 2; grid-column: 1; border-top: 1px solid var(--line); padding: 8px 14px 10px;
                background: linear-gradient(180deg, #0a121c, var(--bg)); }
    #bmsPanel[hidden] { display: none; }
    main { grid-template-rows: 1fr auto; } #stage { grid-row: 1; grid-column: 1; } aside { grid-row: 1 / span 2; grid-column: 2; }
    #bmsPanel .head { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; cursor: pointer; user-select: none; }
    #bmsPanel { --bat: #ffb454; --bat-2: #ffd08a; }
    #bmsPanel .head h2 { margin: 0; font-size: 11px; letter-spacing: .16em; text-transform: uppercase; color: var(--bat); }
    #bmsPanel .head h2::before { content: '▾ '; color: var(--bat); } #bmsPanel.folded .head h2::before { content: '▸ '; }
    #bmsPanel.folded .body { display: none; } #bmsPanel.folded { padding-bottom: 8px; }
    #bmsPanel .head span { font-size: 11px; color: var(--ink-faint); }
    #bmsPanel .row3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
    .bt { background: var(--panel); border: 1px solid #3a3222; border-radius: 10px; padding: 7px 10px; min-width: 0; }
    .bt h3 { margin: 0 0 2px; font-size: 9.5px; text-transform: uppercase; letter-spacing: .12em; color: var(--ink-faint); font-weight: 600; }
    .bt .big { font-size: 21px; font-weight: 600; color: var(--bat-2); font-variant-numeric: tabular-nums; line-height: 1.1; }
    .bt .big small { font-size: 10.5px; color: var(--ink-dim); font-weight: 400; margin-left: 3px; }
    .bt .sub { font-size: 10.5px; color: var(--ink-dim); margin-top: 2px; font-variant-numeric: tabular-nums; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .bt.warn { border-color: var(--warn); } .bt.bad { border-color: var(--bad); } .bt.bad .big { color: var(--bad); }
    .bms-i { cursor: help; color: var(--ink-dim); font-size: 10px; margin-left: 4px; border: 1px solid var(--ink-dim);
             border-radius: 50%; padding: 0 3px; text-transform: none; letter-spacing: 0; }
    #bmsJointsHead { font-size: 10px; text-transform: uppercase; letter-spacing: .12em; color: var(--ink-faint);
                     cursor: pointer; user-select: none; margin-top: 8px; }
    #bmsJoints { display: none; margin-top: 6px; } #bmsJoints.open { display: block; }
    #bmsJoints svg { width: 100%; height: 64px; display: block; }
    #bmsJoints .kv { display: flex; justify-content: space-between; font-size: 11px; color: var(--ink-dim); }
    #bmsJoints .kv b { color: var(--ink); font-weight: 500; font-variant-numeric: tabular-nums; }
  `;
  const INFO = {
    power: 'P = Σ|τ·q̇|/η + copper I²R + 60 W idle — compute draws too (Jetson, boards, LiDAR, camera), not just joint force. MATH.md §2. Real G1: YES — V×I from rt/lf/bmsstate.',
    temp: 'T_pack += dt/C · [I²·R_int (self-heat) − (T_pack − T_outside)/R_th (cold leak)]. Outside temperature pulls the pack down through the jacket (hardcoded 60 % insulation). MATH.md §4+§6. Real G1: YES — BmsState_.temperature[12 sensors].',
    soc: 'SOC −= 100·I·dt / (3600 · 9 Ah · f(T_pack)); a cold pack (f < 1) drains faster. MATH.md §3. Real G1: YES — BmsState_.soc, rt/lf/bmsstate.',
    joints: 'τ = actuator_force, q̇ = actuator_velocity — the only model inputs; everything above is derived. MATH.md §1. Real G1: YES — tau_est / dq in rt/lowstate at 500 Hz (τ is estimated from motor current, no force sensor).',
  };
  const info = k => `<span class="bms-i" title="${INFO[k]}">i</span>`;
  const panel = `
    <div class="head"><h2>Battery Management System</h2><span>simulated · app/bms/sim</span></div>
    <div class="body">
    <div class="row3">
      <div class="bt"><h3>electrical power${info('power')}</h3><div class="big"><span id="bP">—</span><small>W</small></div><div class="sub" id="bPsub">—</div></div>
      <div class="bt" id="btT"><h3>battery pack temperature${info('temp')}</h3><div class="big"><span id="bTbat">—</span><small>°C</small></div><div class="sub" id="bTsub">—</div></div>
      <div class="bt" id="btSoc"><h3>battery life · SOC${info('soc')}</h3><div class="big"><span id="bSoc">—</span><small>%</small></div><div class="sub" id="bTte">—</div></div>
    </div>
    <div id="bmsJointsHead">▸ torque and velocity of joints${info('joints')}</div>
    <div id="bmsJoints" class="bt">
      <div class="kv"><span>max <b id="bTauMax">—</b></span><span>red = max · grey = braking · hover a bar for τ and q̇</span></div>
      <svg id="tauChart" viewBox="0 0 360 64" preserveAspectRatio="none"></svg>
    </div>
    </div>`;
  const knobs = `
    <h2>Battery</h2>
    <div class="row"><label>ambient temperature</label><output id="tambOut">15 °C</output></div>
    <input type="range" id="tamb" min="-30" max="45" step="1" value="15">
    <div class="row"><label>start SOC</label><output id="soc0Out">100 %</output></div>
    <input type="range" id="soc0" min="5" max="100" step="5" value="100">
    <p class="hint">Sliding the temperature cold-soaks pack and motors to it; R<sub>int</sub> doubles every −15 °C. Start SOC restarts the pack.</p>`;

  // ---- mount ----
  const style = document.createElement('style'); style.textContent = css; document.head.appendChild(style);
  const section = document.createElement('section'); section.id = 'bmsPanel'; section.innerHTML = panel;
  document.getElementById('stage').insertAdjacentElement('afterend', section);
  const knobBox = document.createElement('div'); knobBox.innerHTML = knobs;
  const resetBtn = document.getElementById('resetBtn');
  resetBtn.parentElement.insertAdjacentElement('beforebegin', knobBox);
  const tab = document.createElement('div'); tab.className = 'tab active'; tab.id = 'bmsTab'; tab.textContent = 'BMS';
  tab.style.marginLeft = 'auto'; tab.style.marginRight = '12px';
  document.getElementById('status').insertAdjacentElement('beforebegin', tab);
  const $ = id => document.getElementById(id);
  const setOpen = open => { section.classList.toggle('folded', !open); tab.classList.toggle('active', open); };
  tab.onclick = () => setOpen(section.classList.contains('folded'));
  section.querySelector('.head').onclick = () => setOpen(section.classList.contains('folded'));
  $('bmsJointsHead').onclick = e => {
    if (e.target.classList.contains('bms-i')) return;
    const j = $('bmsJoints'); j.classList.toggle('open');
    $('bmsJointsHead').innerHTML = (j.classList.contains('open') ? '▾' : '▸') + ' torque and velocity of joints' + info('joints');
  };
  $('tamb').addEventListener('input', e => { $('tambOut').textContent = e.target.value + ' °C'; send({ type: 'knob', name: 't_amb', value: Number(e.target.value) }); });
  $('soc0').addEventListener('input', e => { $('soc0Out').textContent = e.target.value + ' %'; });
  $('soc0').addEventListener('change', e => send({ type: 'knob', name: 'soc0', value: Number(e.target.value) }));

  // ---- paint ----
  let names = null;
  const fmt = (x, d = 1) => (x === null || x === undefined || !isFinite(x)) ? '—' : Number(x).toFixed(d);
  function paint(b) {
    $('bP').textContent = fmt(b.power_W, 0);
    $('bPsub').textContent = 'mech ' + fmt(b.mech_power_W, 0) + ' W + idle 60 W' + (b.power_avg_W !== undefined ? ' · avg ' + fmt(b.power_avg_W, 0) : '');
    $('bSoc').textContent = b.bms_cutoff ? 'OFF' : fmt(b.soc_pct, 1);
    $('bTte').textContent = b.bms_cutoff ? 'cut-off · ' + fmt(b.soc_pct, 0) + '% left in the cells' : (b.time_to_empty_min == null ? '∞' : fmt(b.time_to_empty_min, 0) + ' min') + ' to empty';
    $('bTbat').textContent = fmt(b.t_bat_C, 1);
    $('bTsub').textContent = 'outside ' + fmt(b.t_amb_C, 0) + ' °C · capacity f ' + fmt(b.capacity_factor, 2);
    $('btSoc').className = 'bt' + (b.bms_cutoff ? ' bad' : b.soc_pct < 20 ? ' warn' : '');
    $('btT').className = 'bt' + (b.t_bat_C > 55 ? ' bad' : b.t_bat_C < 0 ? ' warn' : '');
    if (b.tau_Nm) {
      const W = 360, H = 64, n = b.tau_Nm.length, bw = W / n, tauMax = Math.max(40, ...b.tau_Nm.map(Math.abs));
      $('tauChart').innerHTML = b.tau_Nm.map((tau, i) => {
        const h = Math.abs(tau) / tauMax * (H - 4), c = i === b.max_abs_tau_joint ? 'var(--bad)' : b.joint_power_W[i] < 0 ? 'var(--ink-faint)' : 'var(--accent)';
        return `<rect x="${i * bw + 1}" y="${H - 2 - h}" width="${bw - 2}" height="${h}" fill="${c}"><title>${names ? names[i] : i}: τ ${tau.toFixed(1)} N·m · q̇ ${b.dq_radps[i].toFixed(2)} rad/s · ${b.joint_power_W[i].toFixed(1)} W</title></rect>`;
      }).join('');
      $('bTauMax').textContent = fmt(b.max_abs_tau_Nm, 1) + ' N·m · ' + (names ? names[b.max_abs_tau_joint] : '#' + b.max_abs_tau_joint);
    }
  }

  // ---- hook the page: live frames carry `bms`; replay frames are rebuilt from hud.json ----
  const original = window.paintHud;
  window.paintHud = function (frame) {
    original(frame);
    if (frame.actuator_names) names = frame.actuator_names;
    if (frame.bms) paint(frame.bms);
    else if (state.mode === 'replay' && state.hud && state.hud.bms_soc_pct) {
      const hud = state.hud, tick = Math.min(hud.time_seconds.length - 1, Math.floor(video.currentTime * Number(hud.control_hz || 50)));
      const at = k => hud[k] ? hud[k][tick] : undefined;
      paint({ soc_pct: at('bms_soc_pct'), power_W: at('bms_power_W'), mech_power_W: at('bms_mech_power_W'),
              t_bat_C: at('bms_t_bat_C'), max_abs_tau_Nm: at('bms_max_abs_tau_Nm') });
    }
  };
})();

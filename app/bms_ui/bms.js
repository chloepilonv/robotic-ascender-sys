// app/bms_ui/bms.js — BATTERY MANAGEMENT SYSTEM panel for app/web/index.html.
// Self-contained: injects its own CSS, panel and sidebar knobs, and hooks the
// page by wrapping paintHud(). Remove the <script> tag and nothing else changes.
// Data comes from app/bms_ui/bridge.py via the websocket `state` message (live)
// and from hud.json `bms_*` arrays (replay).
(function () {
  const HISTORY_SECONDS = 120;
  const css = `
    #bmsPanel { grid-row: 2; grid-column: 1; border-top: 1px solid var(--line); padding: 8px 14px 10px;
                background: linear-gradient(180deg, #0a121c, var(--bg)); }
    #bmsPanel[hidden] { display: none; }
    main { grid-template-rows: 1fr auto; } #stage { grid-row: 1; grid-column: 1; } aside { grid-row: 1 / span 2; grid-column: 2; }
    #bmsPanel .head { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
    #bmsPanel { --bat: #ffb454; --bat-2: #ffd08a; }
    #bmsPanel .head { cursor: pointer; user-select: none; }
    #bmsPanel .head h2 { margin: 0; font-size: 11px; letter-spacing: .16em; text-transform: uppercase; color: var(--bat); }
    #bmsPanel .head h2::before { content: '▾ '; color: var(--bat); } #bmsPanel.folded .head h2::before { content: '▸ '; }
    #bmsPanel.folded .body { display: none; } #bmsPanel.folded { padding-bottom: 8px; }
    #bmsPanel.folded #bmsMore { visibility: hidden; }
    #bmsPanel .head span { font-size: 11px; color: var(--ink-faint); }
    #bmsPanel .head .lnk { margin-left: auto; font-size: 11px; color: var(--ink-dim); cursor: pointer; user-select: none; }
    #bmsPanel .head .lnk:hover { color: var(--accent-2); }
    #bmsPanel .row6 { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)) 220px; gap: 8px; }
    .bt { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 7px 10px; min-width: 0; }
    .bt h3 { margin: 0 0 2px; font-size: 9.5px; text-transform: uppercase; letter-spacing: .12em; color: var(--ink-faint); font-weight: 600; }
    .bt .big { font-size: 21px; font-weight: 600; color: var(--bat-2); font-variant-numeric: tabular-nums; line-height: 1.1; }
    .bt .big small { font-size: 10.5px; color: var(--ink-dim); font-weight: 400; margin-left: 3px; }
    .bt .sub { font-size: 10.5px; color: var(--ink-dim); margin-top: 2px; font-variant-numeric: tabular-nums; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .bt { border-color: #3a3222; } .bt.warn { border-color: var(--warn); } .bt.bad { border-color: var(--bad); } .bt.bad .big { color: var(--bad); }
    .bt svg { width: 100%; height: 46px; display: block; }
    #bmsDetails { display: none; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; }
    #bmsDetails.open { display: grid; }
    #bmsDetails svg { height: 64px; }
    #bmsDetails .kv { display: flex; justify-content: space-between; font-size: 11px; color: var(--ink-dim); }
    #bmsDetails .kv b { color: var(--ink); font-weight: 500; font-variant-numeric: tabular-nums; }
  `;
  const panel = `
    <div class="head"><h2>Battery Management System</h2><span>simulated · app/bms/sim</span>
      <span class="lnk" id="bmsMore">details ▸</span></div>
    <div class="body">
    <div class="row6">
      <div class="bt" id="btSoc"><h3>charge</h3><div class="big"><span id="bSoc">—</span><small>%</small></div><div class="sub" id="bTte">—</div></div>
      <div class="bt" id="btV"><h3>pack</h3><div class="big"><span id="bV">—</span><small>V</small></div><div class="sub" id="bSag">—</div></div>
      <div class="bt"><h3>current</h3><div class="big"><span id="bI">—</span><small>A</small></div><div class="sub" id="bWh">—</div></div>
      <div class="bt"><h3>power</h3><div class="big"><span id="bP">—</span><small>W</small></div><div class="sub" id="bPsub">—</div></div>
      <div class="bt" id="btT"><h3>pack temp</h3><div class="big"><span id="bTbat">—</span><small>°C</small></div><div class="sub" id="bTsub">—</div></div>
      <div class="bt"><h3>R internal</h3><div class="big"><span id="bR">—</span><small>mΩ</small></div><div class="sub" id="bRsub">—</div></div>
      <div class="bt"><h3>R_int vs temperature</h3><svg id="rintChart" viewBox="0 0 200 46" preserveAspectRatio="none"></svg></div>
    </div>
    <div id="bmsDetails">
      <div class="bt"><h3>joint torque |τ| · red = max</h3><div class="kv"><span id="bTauMax">—</span><span>grey = braking</span></div>
        <svg id="tauChart" viewBox="0 0 360 64" preserveAspectRatio="none"></svg></div>
      <div class="bt"><h3>last ${HISTORY_SECONDS} s · SOC (blue) · V (amber) · P (green)</h3>
        <div class="kv"><span>P<sub>mech</sub> Σ|τ·q̇| <b id="bPmech">—</b></span><span>copper <b id="bPcu">—</b></span><span>η <b id="bEff">—</b></span><span>hottest <b id="bTmot">—</b></span></div>
        <svg id="histChart" viewBox="0 0 360 64" preserveAspectRatio="none"></svg></div>
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
  section.querySelector('.head').onclick = e => { if (e.target.id !== 'bmsMore') setOpen(section.classList.contains('folded')); };
  $('bmsMore').onclick = () => { const d = $('bmsDetails'); d.classList.toggle('open'); $('bmsMore').textContent = d.classList.contains('open') ? 'details ▾' : 'details ▸'; };
  $('tamb').addEventListener('input', e => { $('tambOut').textContent = e.target.value + ' °C'; send({ type: 'knob', name: 't_amb', value: Number(e.target.value) }); });
  $('soc0').addEventListener('input', e => { $('soc0Out').textContent = e.target.value + ' %'; });
  $('soc0').addEventListener('change', e => send({ type: 'knob', name: 'soc0', value: Number(e.target.value) }));

  // ---- paint ----
  const history = []; let rintCurve = null, names = null;
  const fmt = (x, d = 1) => (x === null || x === undefined || !isFinite(x)) ? '—' : Number(x).toFixed(d);
  const line = (pts, c, w = 1.5) => `<polyline fill="none" stroke="${c}" stroke-width="${w}" points="${pts.map(p => p.join(',')).join(' ')}"/>`;
  function paint(b) {
    $('bSoc').textContent = b.bms_cutoff ? 'OFF' : fmt(b.soc_pct, 1);
    $('bTte').textContent = b.bms_cutoff ? 'cut-off · ' + fmt(b.soc_pct, 0) + '% left in the cells' : (b.time_to_empty_min == null ? '∞' : fmt(b.time_to_empty_min, 0) + ' min') + ' to empty';
    $('bV').textContent = fmt(b.pack_V, 2); $('bSag').textContent = 'sag ' + fmt(b.v_ocv_V - b.pack_V, 2) + ' V';
    $('bI').textContent = fmt(b.current_A, 2); $('bWh').textContent = fmt(b.energy_used_Wh, 2) + ' Wh used';
    $('bP').textContent = fmt(b.power_W, 0); $('bPsub').textContent = 'mech ' + fmt(b.mech_power_W, 0) + ' W' + (b.power_avg_W !== undefined ? ' · avg ' + fmt(b.power_avg_W, 0) : '');
    $('bTbat').textContent = fmt(b.t_bat_C, 1); $('bTsub').textContent = 'ambient ' + fmt(b.t_amb_C, 0) + ' °C · f ' + fmt(b.capacity_factor, 2);
    $('bR').textContent = fmt(b.r_int_ohm * 1000, 0); $('bRsub').textContent = '80 mΩ at 25 °C';
    $('btSoc').className = 'bt' + (b.bms_cutoff ? ' bad' : b.soc_pct < 20 ? ' warn' : '');
    $('btT').className = 'bt' + (b.t_bat_C > 55 ? ' bad' : b.t_bat_C < 0 ? ' warn' : '');
    $('btV').className = 'bt' + (b.pack_V < 40 ? ' warn' : '');
    if (rintCurve) {
      const W = 200, H = 46, t0 = rintCurve[0][0], t1 = rintCurve[rintCurve.length - 1][0], rMax = rintCurve[0][1];
      const X = t => 4 + (t - t0) / (t1 - t0) * (W - 8), Y = r => H - 4 - r / rMax * (H - 10);
      const tb = Math.max(t0, Math.min(t1, b.t_bat_C));
      $('rintChart').innerHTML = line(rintCurve.map(([t, r]) => [X(t), Y(r)]), 'var(--bat)') +
        `<circle cx="${X(tb)}" cy="${Y(b.r_int_ohm)}" r="3.5" fill="#fff"/>` +
        `<text x="${X(t0)}" y="${H - 1}" font-size="7" fill="var(--ink-faint)">${t0}°</text><text x="${X(t1) - 14}" y="${H - 1}" font-size="7" fill="var(--ink-faint)">${t1}°</text>`;
    }
    if (b.tau_Nm) {
      const W = 360, H = 64, n = b.tau_Nm.length, bw = W / n, tauMax = Math.max(40, ...b.tau_Nm.map(Math.abs));
      $('tauChart').innerHTML = b.tau_Nm.map((tau, i) => {
        const h = Math.abs(tau) / tauMax * (H - 4), c = i === b.max_abs_tau_joint ? 'var(--bad)' : b.joint_power_W[i] < 0 ? 'var(--ink-faint)' : 'var(--accent)';
        return `<rect x="${i * bw + 1}" y="${H - 2 - h}" width="${bw - 2}" height="${h}" fill="${c}"><title>${names ? names[i] : i}: ${tau.toFixed(1)} N·m · ${b.joint_power_W[i].toFixed(1)} W</title></rect>`;
      }).join('');
      $('bTauMax').textContent = fmt(b.max_abs_tau_Nm, 1) + ' N·m · ' + (names ? names[b.max_abs_tau_joint] : '#' + b.max_abs_tau_joint);
      $('bPmech').textContent = fmt(b.mech_power_W, 0) + ' W'; $('bPcu').textContent = fmt(b.copper_loss_W, 0) + ' W';
      $('bEff').textContent = fmt(b.efficiency, 2); $('bTmot').textContent = fmt(b.max_temp_winding_C, 0) + ' °C ' + (names ? names[b.hot_joint] : '');
    }
    const t = Number(b._t || 0), last = history[history.length - 1];
    if (last && t < last.t) history.length = 0;
    if (!history.length || t > history[history.length - 1].t + 0.2) history.push({ t, soc: b.soc_pct, v: b.pack_V, p: b.power_W });
    while (history.length && history[0].t < t - HISTORY_SECONDS) history.shift();
    if (history.length > 1) {
      const W = 360, H = 64, t0 = history[0].t, t1 = Math.max(t0 + 5, t), pMax = Math.max(100, ...history.map(h => h.p));
      const X = tt => 2 + (tt - t0) / (t1 - t0) * (W - 4);
      $('histChart').innerHTML = line(history.map(h => [X(h.t), H - 2 - h.p / pMax * (H - 4)]), 'var(--good)', 1) +
        line(history.map(h => [X(h.t), H - 2 - (h.v - 36) / 20 * (H - 4)]), 'var(--warn)') +
        line(history.map(h => [X(h.t), H - 2 - h.soc / 100 * (H - 4)]), 'var(--accent)', 2);
    }
  }

  // ---- hook the page: live frames carry `bms`; replay frames are rebuilt from hud.json ----
  const original = window.paintHud;
  window.paintHud = function (frame) {
    original(frame);
    if (frame.r_int_curve) rintCurve = frame.r_int_curve;
    if (frame.actuator_names) names = frame.actuator_names;
    if (frame.bms) paint(Object.assign({ _t: frame.time_seconds }, frame.bms));
    else if (state.mode === 'replay' && state.hud && state.hud.bms_soc_pct) {
      const hud = state.hud, tick = Math.min(hud.time_seconds.length - 1, Math.floor(video.currentTime * Number(hud.control_hz || 50)));
      const at = k => hud[k] ? hud[k][tick] : undefined;
      paint({ _t: hud.time_seconds[tick], soc_pct: at('bms_soc_pct'), pack_V: at('bms_pack_V'), current_A: at('bms_current_A'),
              power_W: at('bms_power_W'), mech_power_W: at('bms_mech_power_W'), t_bat_C: at('bms_t_bat_C'), r_int_ohm: at('bms_r_int_ohm'),
              energy_used_Wh: at('bms_energy_used_Wh'), max_abs_tau_Nm: at('bms_max_abs_tau_Nm'), v_ocv_V: 13 * (3 + 1.2 * at('bms_soc_pct') / 100) });
    }
  };
})();

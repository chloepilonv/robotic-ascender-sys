# MATH — every monitored value, its source and the equation

"Base" = the robot root link (pelvis). Base IMU = inertial sensor on it; base pose = its world position + orientation.
Units: SI unless stated. Joint vectors are in SDK order (29 joints).

## 1. Raw — read directly
| Value | Symbol | REAL source (`unitree_hg`) | SIM source (MuJoCo `mjData`) | Unit | Rate |
|---|---|---|---|---|---|
| joint position | q | `motor_state[i].q` | `qpos[7:]` | rad | 500 Hz / sim dt |
| joint velocity | q̇ | `motor_state[i].dq` | `qvel[6:]` | rad/s | |
| joint acceleration | q̈ | `motor_state[i].ddq` | `qacc[6:]` | rad/s² | |
| joint torque | τ | `motor_state[i].tau_est` | `actuator_force` | N·m | |
| motor temp (winding / board) | T_m | `motor_state[i].temperature[0]/[1]` | thermal model (§4) | °C | |
| motor bus voltage | V_bus | `motor_state[i].vol` | = V_pack | V | |
| base angular velocity | ω | `imu_state.gyroscope` | gyro sensor | rad/s | |
| base linear acceleration | a | `imu_state.accelerometer` | accelerometer sensor | m/s² | |
| base orientation | q_wxyz / rpy | `imu_state.quaternion / rpy` | `qpos[3:7]` | –, rad | |
| base position / velocity | p, v | – (estimate only) | `qpos[0:3]`, `qvel[0:3]` | m, m/s | sim only |
| foot contact force | F_c | – | `cfrc_ext` / `contact` | N | sim only |
| state of charge | SOC | `BmsState_.soc` | model (§3) | % | low freq |
| state of health | SOH | `BmsState_.soh` | const 100 | % | |
| pack voltage | V_pack | `BmsState_.bmsvoltage[0]` / 1000 | model (§3) | V | |
| pack current | I | `BmsState_.current` / 1000 | model (§3) | A (+ = discharge) | |
| cell voltages | V_cell,k | `BmsState_.cell_vol[k]` / 1000 | V_pack / 13 | V | |
| battery temperature | T_bat | `BmsState_.temperature[]` (min used) | thermal model (§4) | °C | |
| cycles | – | `BmsState_.cycle` | const | count | |
| mainboard temps / fans | – | `MainBoardState_.temperature[] / fan_state[]` | – | °C, rpm | low freq |

## 2. Derived — motion & power (`derived.py`, identical in REAL and SIM)
| Value | Math | Unit |
|---|---|---|
| mechanical power | P_mech = Σ_i \|τ_i · q̇_i\| | W |
| copper loss | P_cu = Σ_i τ_i² · R / k_t²   (R = winding resistance Ω, k_t = torque const N·m/A) | W |
| electrical power (SIM) | P_elec = P_mech / η + P_cu + P_idle   (η ≈ 0.7, P_idle ≈ 60 W: Jetson + boards + LiDAR) | W |
| electrical power (REAL) | P_elec = V_pack · I | W |
| 60 s average power | P_avg = mean(P_elec over last 60 s) | W |
| drivetrain efficiency | η_meas = P_mech / P_elec | – |
| energy used | E = ∫ P_elec dt | Wh |
| max torque joint | argmax_i \|τ_i\| | index |
| hottest joint | argmax_i T_m,i | index |
| cell spread | ΔV = max_k V_cell,k − min_k V_cell,k   (> 50 mV = unbalanced) | mV |
| cost of transport | CoT = P_elec / (m · g · v)   (m = 35 kg, v = base forward speed) | – |
| walking speed | v = ‖v_xy‖ (sim) or ∫a with drift (real) | m/s |
| fall detection | \|pitch\| or \|roll\| > 60° | bool |

## 3. Battery model (SIM; REAL reads SOC/V/I from the BMS)
| Value | Math | Unit |
|---|---|---|
| pack current | I = P_elec / V_pack | A |
| capacity factor | f(T_bat) = clip(1 − 0.0085·(25 − T_bat), 0.4, 1) for T < 25 °C, else 1; f = 0 below −20 °C | – |
| state of charge | SOC(t) = SOC₀ − 100 · ∫ I dt / (3600 · C₂₅ · f(T_bat))   (C₂₅ = 9 Ah) | % |
| open-circuit voltage | V_ocv(SOC) = 13 · (3.0 + 1.2 · SOC/100)   (linear 13S approx, 39–54.6 V) | V |
| internal resistance | R_int(T) = R₂₅ · 2^((25 − T_bat)/15)   (R₂₅ ≈ 0.08 Ω; doubles every −15 °C) | Ω |
| pack voltage under load | V_pack = V_ocv(SOC) − I · R_int(T_bat) | V |
| BMS cut-off | V_pack < 13 · 3.0 V → discharge stops (this is why cold "empties" the pack early) | bool |
| time to empty | TTE = SOC/100 · C₂₅ · V_nom · f(T_bat) / P_avg   (V_nom = 48 V) | min |

## 4. Thermal models (SIM only; REAL reads sensors)
First-order lumped model, integrated with the sim dt:
| Value | Math | Unit |
|---|---|---|
| motor winding | C_th · dT_m/dt = τ² R / k_t² − (T_m − T_amb) / R_th   (C_th ≈ 50 J/K, R_th ≈ 2 K/W) | °C |
| battery | C_bat · dT_bat/dt = I² R_int − (T_bat − T_amb) / R_th,bat   (C_bat ≈ 2000 J/K, R_th,bat ≈ 1.5 K/W) | °C |
| over-temperature flag | T_m > 80 °C or T_bat > 55 °C | bool |

## 5. Environment (SIM inputs; altitude, wind are the knobs)
| Value | Math | Unit |
|---|---|---|
| ambient temperature | T_amb = T_sea − 6.5 °C/km · h   (ISA lapse rate; h = altitude) | °C |
| wind chill (exposed parts) | T_wc = 13.12 + 0.6215·T_amb − 11.37·v_w^0.16 + 0.3965·T_amb·v_w^0.16   (v_w in km/h) | °C |
| air density | ρ = 1.225 · exp(−h / 8400 m) | kg/m³ |
| cooling loss at altitude | R_th ← R_th · (1.225/ρ)^0.5   (thinner air = worse convection) | K/W |
| snow/ice friction | μ ≈ 0.05 (ice, −5 °C) … 0.3 (wet snow) → MuJoCo `geom_friction` | – |

Loop: altitude → T_amb, ρ → T_bat, T_m → f(T), R_int → SOC, V_pack, TTE; friction → slips → τ ↑ → P_elec, heat ↑.

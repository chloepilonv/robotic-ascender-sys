# Environments

| Path | What |
|---|---|
| `himalaya_3m/` | 10 m x 10 m test pad USD (snow / ice / 10° / 40° / wall) + builder + Isaac Lab cfg |
| `himalaya_scene.py` | **Isaac Sim API script** that assembles pad + `robots/g1_unitree.usd` + lights + physics |
| `lhotse_face/` | **MuJoCo** hfield of the real Lhotse Face, Camp II→III (6907 m, 38.9°) + fixed-rope waypoints |

```bash
# in the Isaac Sim container (streams to the web viewer on :8210)
sudo docker exec -it isim-isaac-sim-1 ./python.sh /workspace/assets/environments/himalaya_scene.py --stream
# smoke test, no GUI
sudo docker exec -it isim-isaac-sim-1 ./python.sh /workspace/assets/environments/himalaya_scene.py --headless --frames 300
```

## `lhotse_face/` — real Everest terrain (MuJoCo)

```bash
cd assets/environments/lhotse_face
python mujoco_scene.py              # viewer
python mujoco_scene.py --headless   # physics check
```

25 x 15 m patch of the South Col route, 90% of the way from Camp 2S to Camp 3S.
Copernicus GLO-30 + OpenStreetMap camp nodes. **Location and 38.9° slope are
real; all detail finer than ~30 m is synthetic** — see its README before
quoting the terrain anywhere.

# Environments

| Path | What |
|---|---|
| `himalaya_3m/` | 10 m x 10 m test pad USD (snow / ice / 10° / 40° / wall) + builder + Isaac Lab cfg |
| `himalaya_scene.py` | **Isaac Sim API script** that assembles pad + `robots/g1_unitree.usd` + lights + physics |

```bash
# in the Isaac Sim container (streams to the web viewer on :8210)
sudo docker exec -it isim-isaac-sim-1 ./python.sh /workspace/assets/environments/himalaya_scene.py --stream
# smoke test, no GUI
sudo docker exec -it isim-isaac-sim-1 ./python.sh /workspace/assets/environments/himalaya_scene.py --headless --frames 300
```

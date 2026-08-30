"""Websocket bridge between the runtime and app/web/index.html (live mode).

Copied from pemba_bench/bench/server.py and adapted to this repo: the static
root is the REPOSITORY root (so the page lives at
http://<host>:<port+1>/app/web/index.html and the episode media at
/app/harness/episodes/...), the world list is a single generated entry for the
team environment, and the two directory listings the page used to scrape are
now explicit JSON endpoints.

Out (binary)  : one JPEG per control tick, 640x480, quality 80.
Out (text)    : {"type":"state", "tick":int, "time_seconds":f,
                 "command":[lin_vel_x, lin_vel_y, ang_vel_yaw],
                 "wind_velocity_world_meters_per_second":[f,f],
                 "wind_force_world_newtons":[f,f,f],
                 "wind_in_training":false, "rope_enabled":bool,
                 "world":str, "world_label":str, "loading":bool,
                 "fell":bool, "fall_reason":str|null,
                 "root_position_world":[f,f,f],
                 "rope_travel_meters":f, "climb_meters":f,
                 "hand_height_on_line_meters":f, "hand_line_error_meters":f,
                 "height_gained_meters":f, "rope_force_newtons":f,
                 "slope_degrees":f, "realtime_factor":f, "heading_degrees":f,
                 "paused":bool}
                A state with "loading":true is the last frame before the loop
                blocks to build a newly selected world; frames resume when it
                lands.
In  (text)    : {"type":"input", "keys":["w"],
                 "camera":{"azimuth_degrees":f, "elevation_degrees":f}}
                    keys holds "w" while the key is down; W is the only key. The
                    camera is a third-person orbit around the pelvis, MuJoCo's own
                    azimuth/elevation convention, defaults 180 / -15. The camera's
                    viewing direction is ALSO the steering input: the robot turns
                    to face it, W walks (climbs) that way.
                {"type":"knob", "name":"wind_x"|"wind_y"|"friction"|"t_amb"|"soc0", "value":f}
                    wind_x / wind_y are the world-frame XY WIND VELOCITY in m/s
                    (NOT newtons -- the force is the quadratic drag law from
                    rl/environment/wind_env.py). friction is the foot geoms' mu.
                {"type":"reset"}      respawn at the knees_bent keyframe, ascender
                                      travel back to 0.
                {"type":"pause", "value":true|false}
                {"type":"world", "name":"climb_30"|"free_30"|"free_0"|"climb_0"}
                    switch maps: the runtime finalises the episode, opens that
                    world (building its model on first selection, ~1.6 s warm,
                    then cached) and starts a new episode folder. Unknown names are
                    ignored with a line on stdout. See app/harness/worlds.py.

HTTP: static files from the repository root, plus two generated endpoints so no
listing can go stale -- GET /api/worlds and GET /api/episodes.
"""
import asyncio
import functools
import http.server
import json
import os
import threading

import websockets

REPOSITORY_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
EPISODES_DIRECTORY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "episodes"
)
EPISODES_URL_PREFIX = "/app/harness/episodes"


class Server:
    def __init__(self, port: int = 8765, worlds=None):
        self.port = port
        self.clients = set()
        self.latest_input = {"keys": [], "camera": {}}
        # wind_x / wind_y are metres per second, not newtons; see the docstring.
        self.knobs = {"wind_x": 0.0, "wind_y": 0.0, "friction": 0.8,
                      "t_amb": 15.0, "soc0": 100.0}   # t_amb in C, soc0 in %
        self.reset_requested = False
        self.paused = False
        self.world_requested = None
        self.worlds = list(worlds or [])
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()
        threading.Thread(target=self._serve_files, daemon=True).start()
        self.ready.wait(timeout=10.0)
        print(f"[server] websocket ws://0.0.0.0:{port}"
              f"   page http://localhost:{port + 1}/app/web/index.html", flush=True)

    def _run(self):
        # websockets >= 14 dropped the legacy awaitable form of
        # `websockets.serve`; the modern one is an async context manager that
        # calls asyncio.get_running_loop(), which raises from a bare thread. So
        # we run one coroutine that opens the server and then parks forever.
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._serve_forever())

    async def _serve_forever(self):
        async with websockets.serve(self._handle, "0.0.0.0", self.port, max_size=None):
            self.ready.set()
            await asyncio.Future()

    def _serve_files(self):
        handler = functools.partial(
            _FileHandler, directory=REPOSITORY_ROOT, server_state=self
        )
        http.server.ThreadingHTTPServer(("0.0.0.0", self.port + 1), handler).serve_forever()

    async def _handle(self, websocket):
        self.clients.add(websocket)
        try:
            async for message in websocket:
                data = json.loads(message)
                if data["type"] == "input":
                    self.latest_input = data
                elif data["type"] == "knob":
                    self.knobs[data["name"]] = float(data["value"])
                elif data["type"] == "reset":
                    self.reset_requested = True
                elif data["type"] == "pause":
                    self.paused = bool(data["value"])
                elif data["type"] == "world":
                    self.world_requested = str(data["name"])
        finally:
            self.clients.discard(websocket)

    def broadcast(self, payload) -> None:
        """payload: bytes (JPEG) or dict (state). Non-blocking; drops if no client."""
        if not self.clients:
            return
        message = payload if isinstance(payload, bytes) else json.dumps(payload)
        for client in list(self.clients):
            asyncio.run_coroutine_threadsafe(client.send(message), self.loop)


def list_episodes() -> list:
    """Every folder under app/harness/episodes/ holding a header.json, newest first."""
    if not os.path.isdir(EPISODES_DIRECTORY):
        return []
    episodes = []
    for name in sorted(os.listdir(EPISODES_DIRECTORY), reverse=True):
        if os.path.isfile(os.path.join(EPISODES_DIRECTORY, name, "header.json")):
            episodes.append({"name": name, "url": f"{EPISODES_URL_PREFIX}/{name}"})
    return episodes


class _FileHandler(http.server.SimpleHTTPRequestHandler):
    """Static files from the repo root, plus the two generated endpoints."""

    def __init__(self, *arguments, server_state=None, **keyword_arguments):
        self.server_state = server_state
        super().__init__(*arguments, **keyword_arguments)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/worlds":
            return self._json({"worlds": self.server_state.worlds})
        if path == "/api/episodes":
            return self._json({"episodes": list_episodes()})
        super().do_GET()

    def _json(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *arguments, **keyword_arguments):
        pass

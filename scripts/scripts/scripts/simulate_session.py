"""
End-to-end smoke test WITHOUT the frontend: registers a lecturer + student, creates a
session, grants consent, streams synthetic two-party audio over the telemetry
WebSockets, prints the live metrics, ends the session and prints the report.

    pip install httpx websockets
    python -m scripts.simulate_session --base http://localhost:8000 --invite CODE_IF_YOU_SET_ONE

Note: synthetic tones are not speech, so ASR will return nothing; timing metrics
(latency, interruptions, talk ratio, synchronisation) work exactly as with real voices.
"""
import argparse
import asyncio
import json

import httpx
import numpy as np
import websockets

SR = 16_000
rng = np.random.default_rng(1)


def synth(duration, intervals):
    n = int(duration * SR)
    t = np.arange(n) / SR
    x = rng.normal(0, 0.002, n)
    voice = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate([140, 280, 420, 560], 1))
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 4 * t)
    mask = np.zeros(n)
    for a, b in intervals:
        mask[int(a * SR):int(b * SR)] = 1.0
    return (np.clip(x + 0.08 * voice * env * mask, -1, 1) * 32767).astype("<i2").tobytes()


async def stream(ws, pcm):
    for i in range(0, len(pcm), 3200):            # 100 ms frames, real-time pacing
        await ws.send(pcm[i:i + 3200])
        await asyncio.sleep(0.1)


async def listen(ws):
    async for raw in ws:
        m = json.loads(raw)
        if m.get("type") == "metrics" and int(m["t"]) % 5 == 0:
            print(f"t={m['t']:>5}s conf={m['confidence_score']:>5} eng={m['engagement_score']:>5} "
                  f"dom={m['dominance_score']:>5} ten={m['tension_score']:>5} sync={m['synchronization_score']:>5} "
                  f"lat={m['avg_latency']} talk={m['student_talk_ratio']}%")


async def main(base: str, invite: str, seconds: int):
    pw = "Passw0rd-sim1"
    with httpx.Client(base_url=base, timeout=30) as c:
        for name, email, role in [("Dr Simulated", "sim.lecturer@example.com", "lecturer"),
                                  ("Sam Student", "sim.student@example.com", "student")]:
            c.post("/api/auth/register", json={"full_name": name, "email": email, "password": pw,
                                               "role": role, "invite_code": invite or None})
        lec = c.post("/api/auth/login", json={"email": "sim.lecturer@example.com", "password": pw}).json()
        stu = c.post("/api/auth/login", json={"email": "sim.student@example.com", "password": pw}).json()
        H = lambda t: {"Authorization": f"Bearer {t['access_token']}"}
        r = c.post("/api/sessions", json={"student_id": stu["user"]["id"], "title": "Simulated viva"}, headers=H(lec))
        if r.status_code == 409:
            r = next(x for x in [c.get("/api/sessions", headers=H(lec))] if x.status_code == 200)
            sess = next(s for s in r.json() if s["status"] == "ACTIVE" and s["student_id"] == stu["user"]["id"])
        else:
            r.raise_for_status()
            sess = r.json()
        c.post(f"/api/sessions/{sess['id']}/consent", json={"consent": True}, headers=H(stu)).raise_for_status()

        ws_base = base.replace("http", "ws", 1)
        room = sess["room_id"]
        examiner = synth(seconds, [(1, 3), (16, 18), (seconds - 8, seconds - 6)])
        student = synth(seconds, [(4, 14), (19, seconds - 9), (seconds - 5, seconds - 1)])

        async with websockets.connect(f"{ws_base}/ws/telemetry/{room}?token={lec['access_token']}") as lws, \
                   websockets.connect(f"{ws_base}/ws/telemetry/{room}?token={stu['access_token']}") as sws:
            listener = asyncio.create_task(listen(lws))

            async def vision():
                for _ in range(seconds * 2):
                    await sws.send(json.dumps({"type": "vision", "face_detected": True, "yaw": float(rng.normal(0, 8)),
                                               "pitch": float(rng.normal(0, 6)), "gaze_x": 0.05, "gaze_y": 0.0}))
                    await asyncio.sleep(0.5)

            await asyncio.gather(stream(lws, examiner), stream(sws, student), vision())
            await asyncio.sleep(2)
            end = c.post(f"/api/sessions/{sess['id']}/end", headers=H(lec)).json()
            listener.cancel()

        print("\n=== FINAL METRICS ===")
        print(json.dumps(end.get("metrics"), indent=2))
        rep = c.get(f"/api/sessions/{sess['id']}/report", headers=H(lec)).json()
        print("\n=== INSIGHTS ===")
        for i in rep["insights"]:
            print(f"[{i['level']}] {i['title']}: {i['message']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--invite", default="")
    ap.add_argument("--seconds", type=int, default=40)
    a = ap.parse_args()
    asyncio.run(main(a.base, a.invite, a.seconds))

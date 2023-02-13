import json
import os
import datetime
import time
import requests
import hashlib
from requests_futures.sessions import FuturesSession

print()
environ = os.environ.copy()
for key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "HF_AUTH_TOKEN"]:
    if environ.get(key, None):
        environ[key] = "XXX"
print(environ)
print()


def get_now():
    return round(time.time() * 1000)


SEND_URL = os.getenv("SEND_URL")
if SEND_URL == "":
    SEND_URL = None

SIGN_KEY = os.getenv("SIGN_KEY", "")
if SIGN_KEY == "":
    SIGN_KEY = None

futureSession = FuturesSession()

container_id = os.getenv("CONTAINER_ID")
if not container_id:
    with open("/proc/self/mountinfo") as file:
        line = file.readline().strip()
        while line:
            if "/containers/" in line:
                container_id = line.split("/containers/")[
                    -1
                ]  # Take only text to the right
                container_id = container_id.split("/")[0]  # Take only text to the left
                break
            line = file.readline().strip()


init_used = False


def clearSession(force=False):
    global session
    global init_used

    if init_used or force:
        session = {"_ctime": get_now()}
    else:
        init_used = True


def getTimings():
    timings = {}
    for key in session.keys():
        if key == "_ctime":
            continue
        start = session[key].get("start", None)
        done = session[key].get("done", None)
        if start and done:
            timings.update({key: session[key]["done"] - session[key]["start"]})
        else:
            timings.update({key: -1})
    return timings


def send(type: str, status: str, payload: dict = {}, opts: dict = {}):
    now = get_now()
    send_url = opts.get("SEND_URL", SEND_URL)
    sign_key = opts.get("SIGN_KEY", SIGN_KEY)

    if status == "start":
        session.update({type: {"start": now, "last_time": now}})
    elif status == "done":
        session[type].update({"done": now, "diff": now - session[type]["start"]})
    else:
        session[type]["last_time"] = now

    data = {
        "type": type,
        "status": status,
        "container_id": container_id,
        "time": now,
        "t": now - session["_ctime"],
        "tsl": now - session[type]["last_time"],
        "payload": payload,
    }

    if send_url and sign_key:
        input = json.dumps(data, separators=(",", ":")) + sign_key
        sig = hashlib.md5(input.encode("utf-8")).hexdigest()
        data["sig"] = sig

    print(datetime.datetime.now(), data)

    if send_url:
        futureSession.post(send_url, json=data)

    # try:
    #    requests.post(send_url, json=data)  # , timeout=0.0000000001)
    # except requests.exceptions.ReadTimeout:
    # except requests.exceptions.RequestException as error:
    #    print(error)
    #    pass


clearSession(True)

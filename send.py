import json
import os
import time
import requests
import hashlib
from requests_futures.sessions import FuturesSession

print(os.environ)


def get_now():
    return round(time.time() * 1000)


send_url = os.getenv("SEND_URL")
sign_key = os.getenv("SIGN_KEY")
init_time = get_now()
session = FuturesSession()
last_time = init_time

with open("/proc/self/mountinfo") as file:
    line = file.readline().strip()
    while line:
        if "/docker/containers/" in line:
            container_id = line.split("/docker/containers/")[
                -1
            ]  # Take only text to the right
            container_id = container_id.split("/")[0]  # Take only text to the left
            break
        line = file.readline().strip()


def send(type: str, status: str, payload: dict = {}, init=False):
    global id
    global dest
    global init_time
    global last_time

    now = get_now()

    if init:
        init_time = now

    data = {
        "type": type,
        "status": status,
        "container_id": container_id,
        "time": now,
        "t": now - init_time,
        "tsl": now - last_time,
        "payload": payload,
    }
    last_time = now

    if init:
        data["init"] = True

    if send_url:
        input = json.dumps(data, separators=(",", ":")) + sign_key
        sig = hashlib.md5(input.encode("utf-8")).hexdigest()
        data["sig"] = sig

    print(data)

    if send_url:
        session.post(send_url, json=data)

    # try:
    #    requests.post(send_url, json=data)  # , timeout=0.0000000001)
    # except requests.exceptions.ReadTimeout:
    # except requests.exceptions.RequestException as error:
    #    print(error)
    #    pass

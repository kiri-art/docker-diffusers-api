# CONTRIBUTING

*Tips for development*

1. [General Hints](#general)
1. [Development / Editor Setup](#editors)
    1. [Visual Studio Code (vscode)](#vscode)
1. [Testing](#testing)
1. [Using Buildkit](#buildkit)
1. [Local HTTP(S) Caching Proxy](#caching)
1. [Local S3 Server](#local-s3-server)
1. [Stop on Suspend](#stop-on-suspend)

<a name="general"></a>
## General

1. Run docker with `-it` to make it easier to stop container with `Ctrl-C`.
1. If you get a `CUDA initialization: CUDA unknown error` after suspend,
    just stop the container, `rmmod nvidia_uvm`, and restart.

<a name="editors"></a>
## Editors

<a name="vscode"></a>
### Visual Studio Code (recommended, WIP)

*We're still writing this guide, let us know of any needed improvements*

This repo includes VSCode settings that allow for a) editing inside a docker container, b) tests and coverage (on save)

1. Install from https://code.visualstudio.com/
1. Install [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension.
1. Open your docker-diffusers-api folder, you'll get a popup in the bottom right that a dev container environment was detected, click "reload in container"
1. Look for the "( ) Watch" on status bar and click it so it changes to "( ) XX Coverage"

**Live Development**

1. **Run Task** (either Ctrl-Shift-P and "Run Task", or in Terminals, the Plus ("+") DROPDOWN selector and choose, "Run Task..." at the bottom)
1. Choose **Watching Server**.  Port 8000 will be forwarded.  The server will be reloaded
on every file safe (make sure to give it enough time to fully load before sending another
request, otherwise that request will hang).

<a name="testing"></a>
## Testing

1. **Unit testing**: exists but is sorely lacking for now.  If you use the
recommended editor setup above, it's probably working already.  However:

1. **Integation / E2E**: cover most features used in production.
`pytest -s tests/integration`.
The `-s` is optional but streams stdout so you can follow along.
Add also `-k test_name` to test a specific test.  E2E tests are LONG but you can
greatly reduce subsequent run time by following the steps below for a
[Local HTTP(S) Caching Proxy](#caching) and [Local S3 Server](#local-s3-server).

Docker-Diffusers-API follows Semantic Versioning.  We follow the
[conventional commits](https://www.conventionalcommits.org/en/v1.0.0/)
standard.

* On a commit to `dev`, if all CI tests pass, a new release is made to `:dev` tag.
* On a commit to `main`, if all CI tests pass, a new release with appropriate
major / minor / patch is made, based on appropriate tags in the commit history.

<a name="buildkit"></a>
## Using BuildKit

Buildkit is a docker extension that can really improve build speeds through
caching and parallelization.  You can enable and tweak it by adding:

  `DOCKER_BUILDKIT=1 BUILDKIT_PROGRESS=plain`

vars before `docker build` (the `PROGRESS` var shows much more detailed
build logs, which can be useful, but are much more verbose).  This is
already all setup in the the [build](./build) script.

<a name="caching"></a>
## Local HTTP(S) Caching Proxy

If you're only editing e.g. `app.py`, there's no need to worry about caching
and the docker layers work amazingly.  But, if you're constantly changing
installed packages (apt, `requirements.txt`), `download.py`, etc, it's VERY
helpful to have a local cache:

```bash
# See all options at https://hub.docker.com/r/gadicc/squid-ssl-zero
$ docker run -d -p 3128:3128 -p 3129:80 \
  --name squid --restart=always \
  -v /usr/local/squid:/usr/local/squid \
  gadicc/squid-ssl-zero
```

and then set the docker build args `proxy=1`, and `http_proxy` / `https_proxy`
with their respective values.
This is already all set up in the [build](./build) script.

**You probably want to fine-tune /usr/local/squid/etc/squid.conf**.

It will be created after you first run `gadicc/squid-ssl-zero`.  You can then
stop the container (`docker ps`, `docker stop container_id`), edit the file,
and re-start (`docker start container_id`).  For now, try something like:

```conf
cache_dir ufs /usr/local/squid/cache 50000 16 256 # 50GB
maximum_object_size 20 GB
refresh_pattern .  52034400 50% 52034400 store-stale override-expire ignore-no-cache ignore-no-store ignore-private
```

but ideally we can as a community create some rules that don't so
aggressively catch every single request.

<a name="local-s3"></a>
## Local S3 server

If you're doing development around the S3 handling, it can be very useful to
have a local S3 server, especially due to the large size of models.  You
can set one up like this:

```bash
$ docker run -p 9000:9000 -p 9001:9001 \
  -v /usr/local/minio:/data quay.io/minio/minio \
  server /data --console-address ":9001"
```

Now point a web browser to http://localhost:9001/, login with the default
root credentials `minioadmin:minioadmin` and create a bucket and credentials
for testing.  More info at https://hub.docker.com/r/minio/minio/.

Typical policy:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "VisualEditor0",
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject"
            ],
            "Resource": "arn:aws:s3:::BUCKET_NAME/*"
        }
    ]
}
```

Then set the **build-arg** `AWS_S3_ENDPOINT_URL="http://172.17.0.1:9000"`
or as appropriate if you've changed the default docker network.

<a name="stop-on-suspend"></a>
## Stop on Suspend

Maybe it's just me, but frequently I'll have issues when suspending with
the container running (I guess its a CUDA issue), either a freeze on resume,
or a stuck-forever defunct process.  I found it useful to automatically stop
the container / process on suspend.

I'm running ArchLinux and set up a `systemd` suspend hook as described
[here](https://wiki.archlinux.org/title/Power_management#Sleep_hooks), to
call a script, which contains:

```bash
# Stop a matching docker container
PID=`docker ps -qf ancestor=gadicc/diffusers-api`
if [ ! -z $PID ] ; then
	echo "Stopping diffusers-api pid $PID"
	docker stop $PID
fi

# For a VSCode devcontainer, just kill the watchmedo process.
PID=`docker ps -qf volume=/home/dragon/root-cache`
if [ ! -z $PID ] ; then
	echo "Stopping watchmedo in container $PID"
	docker exec $PID /bin/bash -c 'kill `pidof -sx watchmedo`'
fi
```

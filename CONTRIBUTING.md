# CONTRIBUTING

*Tips for development*

## Using BuildKit

Buildkit is a docker extension that can really improve build speeds through
caching and parallelization.  You can enable and tweak it by adding:

  `DOCKER_BUILDKIT=1 BUILDKIT_PROGRESS=plain`

vars before `docker build` (the `PROGRESS` var shows much more detailed
build logs, which can be useful, but are much more verbose).  This is
already all setup in the the [build](./build) script.

## Cache

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
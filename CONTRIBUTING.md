# CONTRIBUTING

*Tips for development*

1. [Using Buildkit](#buildkit)
1. [Local HTTP(S) Caching Proxy](#caching)
1. [Local S3 Server](#local-s3-server)

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
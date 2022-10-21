# root-cache

Useful during dev to avoid redownloading packages and models.
Only necessary when changing requirements or stable diffusion stuff.
If you're justy modifying `app.py`, docker's layered filesystem
works great.

After you've built once, or anytime new files were downloaded,
you need to resync the cache.  Run the container however you
usually do, and then:

```bash
$ docker ps # to get container id
$ docker cp container_id:/root/.cache tmp
$ rm -rf root-cache
$ mv tmp root-cache
```

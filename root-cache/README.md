# root-cache

Useful during dev to avoid redownloading packages and models.
Only necessary when changing requirements or stable diffusion stuff.

After you've built once, copy the files from the container:

```bash
$ docker cp container_id:/root.cache root-cache
```

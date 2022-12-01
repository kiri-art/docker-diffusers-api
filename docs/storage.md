# Storage

Most URLs passed at build args or call args support special URLs, both to
store and retrieve files.

**The Storage API is new and may change without notice, please keep a
careful look in the CHANGELOG when upgrading**.

* [AWS S3](#s3)

<a name="s3"></a>
## S3

### Build Args

Set the following **build-args**, as appropriate (through the Banana dashboard,
by modifying the appropriate lines in the `Dockerfile`, or by specifying, e.g.
`--build-arg AWS_ACCESS_KEY="XXX"` etc.)

```Dockerfile
ARG AWS_ACCESS_KEY_ID="XXX"
ARG AWS_SECRET_ACCESS_KEY="XXX"
ARG AWS_DEFAULT_REGION="us-west-1" # best for banana
# Optional.  ONLY SET THIS IF YOU KNOW YOU NEED TO.
# Usually only if you're using non-Amazon S3-compatible storage.
# If you need this, your provider will tell you exactly what
# to put here.  Otherwise leave it blank to automatically use
# the correct Amazon S3 endpoint.
ARG AWS_S3_ENDPOINT_URL
```

### Usage

In any URL where Storage is supported (e.g. dreambooth `dest_url`):

  * `s3://endpoint/bucket/path/to/file`
  * `s3:///bucket/file` (uses the default endpoint)
  * `s3:///bucket` (for `dest_url`, filename will match your output model)
  * `http+s3://...` (force http instead of https)
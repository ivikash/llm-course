# LLM Course — Infra (CDK)

Deploys the static course site:

- **S3** bucket, private, Block Public Access on.
- **CloudFront** distribution with OAC and the S3 **regional REST endpoint**
  as origin (`bucket.s3.<region>.amazonaws.com`).
- Everything (the `site/` shell and every lesson `.md`) is uploaded so the
  runtime `fetch('../<path>.md')` calls from `site/index.html` resolve.
- A tiny `/index.html` at the bucket root redirects to `/site/index.html`.

## Deploy

```bash
cd infra
npm install
# once per account/region:
npx cdk bootstrap aws://<YOUR_ACCOUNT_ID>/us-west-2
npx cdk deploy
```

The stack outputs a `SiteUrl` like `https://dXXXX.cloudfront.net/`.

## Update content

Re-run `npx cdk deploy` after editing lessons. `BucketDeployment` hashes the
source tree, uploads changes, and invalidates `/*` on the distribution.

## Tear down

```bash
npx cdk destroy
```

This deletes the bucket (objects included) and the distribution. CloudFront
takes a few minutes to fully drain.

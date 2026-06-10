import * as path from 'path';
import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as cloudfront from 'aws-cdk-lib/aws-cloudfront';
import * as origins from 'aws-cdk-lib/aws-cloudfront-origins';
import * as s3deploy from 'aws-cdk-lib/aws-s3-deployment';

/**
 * LLM Course static site.
 *
 * - Private S3 bucket (Block Public Access on).
 * - CloudFront distribution using the S3 REGIONAL REST endpoint as origin
 *   (via `S3BucketOrigin.withOriginAccessControl`, which wires up OAC and
 *   the bucket policy automatically). This is what "Regional" means in the
 *   modern CloudFront → S3 pattern.
 * - BucketDeployment uploads the entire course directory so that lesson
 *   markdown files are reachable at the paths `site/index.html` fetches
 *   (it does `fetch('../' + path)` from inside `site/`).
 * - A tiny redirect `index.html` at the bucket root sends the apex URL to
 *   `/site/index.html` so visitors land on the app shell.
 */
export class LlmCourseSiteStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // Course content lives one directory up from the infra/ package.
    const courseRoot = path.join(__dirname, '..', '..');

    const bucket = new s3.Bucket(this, 'SiteBucket', {
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      enforceSSL: true,
      versioned: false,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    const distribution = new cloudfront.Distribution(this, 'Distribution', {
      comment: 'LLM Course static site',
      defaultBehavior: {
        origin: origins.S3BucketOrigin.withOriginAccessControl(bucket),
        viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
        cachePolicy: cloudfront.CachePolicy.CACHING_OPTIMIZED,
        compress: true,
        allowedMethods: cloudfront.AllowedMethods.ALLOW_GET_HEAD,
      },
      defaultRootObject: 'index.html',
      priceClass: cloudfront.PriceClass.PRICE_CLASS_100,
      httpVersion: cloudfront.HttpVersion.HTTP2_AND_3,
    });

    // Upload the whole course tree (md files + site shell) plus a root
    // redirect index.html that bounces visitors to /site/index.html.
    new s3deploy.BucketDeployment(this, 'DeployCourse', {
      sources: [
        s3deploy.Source.asset(courseRoot, {
          exclude: [
            'infra/**',
            '.git/**',
            '.github/**',
            'node_modules/**',
            '**/.DS_Store',
            'cdk.out/**',
            '*.log',
          ],
        }),
        s3deploy.Source.data(
          'index.html',
          '<!doctype html><html><head><meta charset="utf-8"><title>LLM Course</title>' +
          '<meta http-equiv="refresh" content="0; url=/site/index.html"></head>' +
          '<body><p>Redirecting to <a href="/site/index.html">/site/index.html</a>…</p></body></html>',
        ),
      ],
      destinationBucket: bucket,
      distribution,
      distributionPaths: ['/*'],
      memoryLimit: 512,
      prune: true,
    });

    new cdk.CfnOutput(this, 'SiteUrl', {
      value: `https://${distribution.distributionDomainName}/`,
      description: 'Open this. The root redirects to /site/index.html.',
    });
    new cdk.CfnOutput(this, 'BucketName', { value: bucket.bucketName });
    new cdk.CfnOutput(this, 'DistributionId', { value: distribution.distributionId });
  }
}

#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { LlmCourseSiteStack } from '../lib/stack';

const app = new cdk.App();

new LlmCourseSiteStack(app, 'LlmCourseSite', {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION ?? 'us-west-2',
  },
  description: 'Static site for the LLM Course — S3 (private) + CloudFront (regional S3 origin + OAC).',
});

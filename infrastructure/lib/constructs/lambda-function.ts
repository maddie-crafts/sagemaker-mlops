import { Construct } from 'constructs';
import * as cdk from 'aws-cdk-lib';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as iam from 'aws-cdk-lib/aws-iam';

interface LambdaFunctionProps {
    functionName: string;
    scriptPath: string;
    targetBucket: s3.IBucket;
}

export class LambdaFunction extends Construct {
  public readonly lambdaFunction: lambda.Function;

  constructor(scope: Construct, id: string, props: LambdaFunctionProps) {
    super(scope, id);

    this.lambdaFunction = new lambda.Function(this, 'LambdaFunction', {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'index.handler',
      code: lambda.Code.fromAsset(props.scriptPath),
      functionName: props.functionName,
      environment: {
        BUCKET_NAME: props.targetBucket.bucketName,
      },
      timeout: cdk.Duration.minutes(3),
      memorySize: 512,
    });
    props.targetBucket.grantRead(this.lambdaFunction);
    props.targetBucket.grantDelete(this.lambdaFunction);
    props.targetBucket.grantWrite(this.lambdaFunction);

    this.lambdaFunction.role?.addManagedPolicy(
      iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSageMakerFullAccess')
    );
  }
}
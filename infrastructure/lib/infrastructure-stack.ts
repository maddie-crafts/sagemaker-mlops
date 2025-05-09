import * as cdk from 'aws-cdk-lib';
import * as path from 'path';
import { Construct } from 'constructs';
import { DefaultVpcConstruct } from './constructs/default-vpc';
import { ClusterConstruct } from './constructs/ecs-cluster';
import { S3Bucket } from './constructs/s3-bucket';
import { LambdaFunction } from './constructs/lambda-function';
import { ECRImageBuildAndTaskDefinition } from './constructs/ecr';

export interface InfrastructureStackProps extends cdk.StackProps {
    branchName: string;
    region: string;
  }
  
  export class InfrastructureStack extends cdk.Stack {
    constructor(scope: Construct, id: string, props?: InfrastructureStackProps) {
      super(scope, id, props);
      const branchName = props?.branchName;
      const codebuildSrcDir = process.env.CODEBUILD_SRC_DIR;
      const projectName = `sagemaker-mlops-${branchName}`;
      const processingBucket = new S3Bucket(this, 'ProcessingBucket', {
          bucketName: `sagemaker-mlops-bucket-${branchName}`,
      });
      const SECRET_NAME = 'database-secrets-in-secrets-manager';
      
      const region = props?.region;

      const defaultVpc = new DefaultVpcConstruct(this, 'DefaultVpc');

      const cluster = new ClusterConstruct(this, 'Cluster', {
        vpc: defaultVpc.vpc,
        clusterName: `${projectName}-cluster`,
      });

      const DeleteModelArtifactsLambda = new LambdaFunction(this, 'DeleteModelArtifacts', {
        functionName: `DeleteModelArtifacts`,
        scriptPath: path.join(__dirname, '../../src/lambda/delete_model_artifacts'),
        targetBucket: processingBucket.bucket,
      }).lambdaFunction;

      const StartRetrainingPipeline = new ECRImageBuildAndTaskDefinition(this, 'StartRetrainingPipeline', {
        localImagePath: path.join(__dirname, '../../src/containers/start_pipeline'),
        taskdefinitionName: `${projectName}-start-training-pipeline`,
        ecrRepositoryName: `start-training-pipeline-${branchName}`,
        cpu: 1024,
        memoryLimit: 3072,
        cluster: cluster.cluster,
        targetBucket: processingBucket.bucket,
        secretName: SECRET_NAME,
        aws_region: `${region}`,
        reuseExistingResources: false,
      }); 

    }
}
# KVS License Plate Detection Demo

This project demonstrates an automated license plate detection system using AWS Kinesis Video Streams (KVS), Amazon Rekognition, Lambda, S3, and DynamoDB.

## Architecture Overview

1. **EC2 Instance**: Runs the KVS producer SDK to stream video
2. **Kinesis Video Stream**: Captures and processes video data
3. **S3 Bucket**: Stores image frames extracted from the video stream
4. **Lambda Function**: Processes images to detect license plates using Rekognition
5. **DynamoDB**: Stores detected license plate information
6. **SNS**: Sends email notification when system is ready to use

## Architecture Diagram
![license-detection-architecture](images/kvs-architecture.png)
    
## Prerequisites

- AWS CLI installed and configured with appropriate permissions
- An AWS account with access to create the required resources
- A valid email address to receive system notifications

## Security Considerations

The CloudFormation template (`aws-kvs-license-detection-demo-cfn.yml`) implements security best practices including:
- **Least Privilege IAM**: Roles have minimal required permissions
- **Resource-Specific Access**: Permissions scoped to specific resources only
- **Encryption**: S3 and DynamoDB encryption enabled
- **IMDSv2**: EC2 instance uses secure metadata service

## Deployment Instructions

### 1. Deploy the CloudFormation Stack

```bash
aws cloudformation create-stack \
  --stack-name kvs-license-detection \
  --template-body file://aws-kvs-license-detection-demo-cfn.yml \
  --parameters ParameterKey=EmailAddress,ParameterValue=your.email@example.com \
  --capabilities CAPABILITY_IAM
```

### 2. Confirm SNS Subscription

1. Check your email for a message from AWS Notifications
2. Click the "Confirm subscription" link in the email
3. You should see a confirmation message in your browser

### 3. Wait for Setup Completion

1. Monitor your email for the "EC2 UserData Script Complete" notification
2. This email will contain:
   - Confirmation that the EC2 instance has finished initialization
   - Direct URL to connect to your EC2 instance console
   - Instance ID and region information
   - AWS Account ID for reference
3. **Important**: 
   - Do not attempt to connect to the EC2 instance until you receive this email
   - Use the provided console URL in the email for easy access to your instance
   - Ensure you're logged into the AWS Console before clicking the URL
4. Setup typically takes 10-15 minutes to complete

Example email content:

```
EC2 instance i-0123456789abcxxxx has completed user-data script initialization.

Instance Details:
- Instance ID: i-0123456789abcxxxx
- Region: us-west-2
- Account ID: 123456789012

Connect to your instance using:
https://123456789012.console.aws.amazon.com/ec2/home?region=us-west-2#ConnectToInstance:instanceId=i-0123456789abcxxxx

Note: Make sure you're logged into the AWS Console before using this URL.
```

### 4. Monitor Stack Creation (Optional)

```bash
aws cloudformation describe-stacks \
  --stack-name kvs-license-detection
```

### 5. Access EC2 Instance

Only proceed with these steps after receiving the completion email:

**Option 1 (Recommended): Use the Session Manager URL from the completion email**
- Click the provided URL directly (ensure you're logged into AWS Console first)

**Option 2: Manual Session Manager Access**
1. Open EC2 Console
2. Look for the running instance named "aws-kvs-demo"
3. Click on Connect button
4. Select **Session Manager** (NOT EC2 Instance Connect)
5. Click Connect

**Note**: This instance uses Session Manager for secure access. No SSH or inbound network access is configured (security best practice). 

## Testing the Solution

### 1. Stream Video to KVS

The EC2 instance automatically runs the streaming service:

```bash
# The service runs automatically, but you can manage it with:
sudo systemctl status kvs-stream.service
```

**Service Management Commands**: 
- Check status: `sudo systemctl status kvs-stream.service`
- Stop streaming: `sudo systemctl stop kvs-stream.service`
- Start streaming: `sudo systemctl start kvs-stream.service`
- Restart streaming: `sudo systemctl restart kvs-stream.service`
- Disable auto-start: `sudo systemctl disable kvs-stream.service`
- Enable auto-start: `sudo systemctl enable kvs-stream.service`

**View Streaming**: AWS Console > Kinesis Video Streams > Video streams > aws-kvs-license-detection-demo > Media playback

### 2. View Results

1. Check the S3 bucket for extracted frames:
   ```bash
   aws s3 ls s3://aws-kvs-license-detection-s3-$(aws sts get-caller-identity --query 'Account' --output text)-$(aws configure get region)
   ```

2. Query DynamoDB for detected license plates:
   ```bash
   aws dynamodb scan --table-name LicensePlateDetections
   ```

## How It Works

1. The EC2 instance streams video to Kinesis Video Stream
2. KVS extracts image frames at 2-second intervals and saves them to S3
3. S3 triggers the Lambda function when new images are uploaded
4. Lambda uses Rekognition to detect license plates in the images
5. If a license plate is detected with high confidence (≥80%), the information is stored in DynamoDB

## Security Features

The template implements:
- **Least Privilege IAM**: Roles have minimal required permissions
- **Resource-Specific Access**: Permissions scoped to specific resources only
- **Encryption**: S3 and DynamoDB encryption enabled
- **IMDSv2**: EC2 instance uses secure metadata service
- **Zero Inbound Access**: Security group has no inbound rules (maximum security)
- **Session Manager Access**: Secure, audited access without SSH exposure

## Reporting Deployment
Please refer to [KVS-Reporting-Deployment](https://github.com/jpsotoal/aws-kvs-license-detection/blob/main/reporting/README.md)

## Cleanup

To delete all resources created by this stack:

```bash
aws cloudformation delete-stack --stack-name kvs-license-detection
```

**Note**: Ensure S3 buckets are empty before deletion, or add `--retain-resources` if needed.

## Troubleshooting

### Setup Phase
- **Missing Confirmation Email**: Check your spam folder for the AWS SNS confirmation email
- **No Completion Email**: The user-data script might have failed. Check EC2 instance logs in CloudWatch
- **Long Setup Time**: The initialization process includes software installation and compilation, which can take 10-15 minutes

### Operation Phase
- **EC2 Connection Issues**: Use Session Manager via AWS Console. No SSH access is configured (security best practice with zero inbound rules)
- **Session Manager Access**: Ensure you're logged into the AWS Console and have appropriate IAM permissions for Session Manager
- **KVS Streaming Issues**: Check EC2 instance logs and ensure IAM permissions are correct
- **Lambda Errors**: Check CloudWatch Logs for the Lambda function
- **Missing License Plate Detections**: Ensure images contain clear, visible license plates for Rekognition to detect

### CloudWatch Logs Access

To check EC2 instance logs:
```bash
# Get instance ID first
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=aws-kvs-demo" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)

# View logs
aws logs describe-log-streams \
  --log-group-name /aws/ec2/user-data \
  --order-by LastEventTime --descending
```

## Support

If you encounter issues:
1. Check CloudWatch Logs for detailed error messages
2. Verify all SNS email confirmations were completed
3. Ensure the user-data script completed successfully (check for completion email)
4. Review IAM roles and permissions if experiencing access issues
5. Review IAM roles and permissions if experiencing access issues

## Cost Optimization

- **EC2**: Consider using Spot instances for development
- **KVS**: Data retention set to 24 hours to minimize costs
- **DynamoDB**: Uses on-demand billing for cost efficiency
- **Lambda**: Reserved concurrency limits prevent runaway costs
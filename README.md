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

## Deployment Instructions

### 1. Deploy the CloudFormation Stack

```bash
aws cloudformation create-stack \
  --stack-name kvs-license-detection \
  --template-body file://aws-kvs-license-detection-demo-cfn.yaml \
  --parameters ParameterKey=EmailAddress,ParameterValue=your.email@example.com \
  --capabilities CAPABILITY_IAM
```

### 2. Confirm SNS Subscription

1. Check your email for a message from AWS Notifications
2. Click the "Confirm subscription" link in the email
3. You should see a confirmation message in your browser

Here's the modified section with the added information about receiving the EC2 console login URL:

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

```markdown
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

1. Open EC2 Console
2. Look for the running instance named "aws-kvs-demo"
3. Click on Connect button
4. Select EC2 Instance Connect
5. Leave the default settings
6. Click Connect 

## Testing the Solution

### 1. Stream Video to KVS

The EC2 instance, will run automatically script below:

```bash
./stream-video.sh

You can check streaming video in: AWS Console > Kinises Video Stream > Video streams > aws-kvs-license-detection-demo > Media playback.
```
**How to**: 
- Check status: `sudo systemctl status kvs-stream.service`

- Stop streaming: `sudo systemctl stop kvs-stream.service`

- Start streaming: `sudo systemctl start kvs-stream.service`

- Restart streaming: `sudo systemctl restart kvs-stream.service`

- Disable auto-start: `sudo systemctl disable kvs-stream.service`

- Enable auto-start: `sudo systemctl enable kvs-stream.service`

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
2. KVS extracts image frames at 5-second intervals and saves them to S3
3. S3 triggers the Lambda function when new images are uploaded
4. Lambda uses Rekognition to detect license plates in the images
5. If a license plate is detected with high confidence, the information is stored in DynamoDB

## Reporting Deployment
Please refer to [KVS-Reporting-Deployment](https://github.com/jpsotoal/aws-kvs-license-detection/blob/main/reporting/README.md)

## Cleanup

To delete all resources created by this stack:

```bash
aws cloudformation delete-stack --stack-name kvs-license-detection
```

## Troubleshooting

### Setup Phase
- **Missing Confirmation Email**: Check your spam folder for the AWS SNS confirmation email
- **No Completion Email**: The user-data script might have failed. Check EC2 instance logs in CloudWatch
- **Long Setup Time**: The initialization process includes software installation and compilation, which can take 10-15 minutes

### Operation Phase
- **EC2 Connection Issues**: Ensure your security group allows SSH access from your IP
- **KVS Streaming Issues**: Check EC2 instance logs and ensure IAM permissions are correct
- **Lambda Errors**: Check CloudWatch Logs for the Lambda function
- **Missing License Plate Detections**: Ensure images contain clear, visible license plates for Rekognition to detect

### CloudWatch Logs Access

To check EC2 instance logs:
```bash
aws logs get-log-events \
  --log-group-name /var/log/cloud-init-output.log \
  --log-stream-name $(curl -s http://169.254.169.254/latest/meta-data/instance-id)
```

## Support

If you encounter issues:
1. Check CloudWatch Logs for detailed error messages
2. Verify all SNS email confirmations were completed
3. Ensure the user-data script completed successfully (check for completion email)
4. Review IAM roles and permissions if experiencing access issues

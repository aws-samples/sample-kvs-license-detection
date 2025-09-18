import json
import urllib.parse
import boto3
import logging
import re
import os
import datetime
from decimal import Decimal
from botocore.exceptions import ClientError

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients (will be initialized in lambda_handler for actual use)
s3 = None
rekognition = None
dynamodb = None

def init_aws_clients():
    """Initialize AWS clients"""
    global s3, rekognition, dynamodb
    if s3 is None:
        s3 = boto3.client('s3')
        rekognition = boto3.client('rekognition')
        dynamodb = boto3.resource('dynamodb')

# Configuration
TABLE_NAME = os.environ.get('LICENSE_PLATE_TABLE', 'LicensePlateDetections')
MIN_CONFIDENCE = 80.0  # Higher confidence for license plates
MIN_TEXT_SIZE = 3      # Minimum characters for license plate
MAX_TEXT_SIZE = 10     # Maximum characters for license plate

def is_license_plate_text(text, confidence):
    """
    Simple check if text looks like a license plate
    Focus on big characters that are typically license plates
    """
    # Clean the text
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    # Basic validation
    if confidence < MIN_CONFIDENCE:
        return False
    
    if len(clean_text) < MIN_TEXT_SIZE or len(clean_text) > MAX_TEXT_SIZE:
        return False
    
    # Filter out common non-license plate words
    common_words = {'STOP', 'YIELD', 'EXIT', 'ENTER', 'PARKING', 'NO', 'YES', 'ONE', 'WAY'}
    if clean_text in common_words:
        return False
    
    # Must contain both letters and numbers (typical license plate pattern)
    has_letters = bool(re.search(r'[A-Z]', clean_text))
    has_numbers = bool(re.search(r'[0-9]', clean_text))
    
    # License plates typically have both letters and numbers
    if has_letters and has_numbers:
        return True
    
    # Or could be a state name (all letters, but longer than common words)
    if has_letters and not has_numbers and len(clean_text) >= 5:
        # Known US states that could appear on license plates
        us_states = {'FLORIDA', 'TEXAS', 'CALIFORNIA', 'NEVADA', 'ARIZONA', 'OREGON', 'WASHINGTON', 'MONTANA', 'ALASKA', 'HAWAII'}
        if clean_text in us_states:
            return True
    
    return False

def extract_license_plates(image_bucket, image_key):
    """
    Extract license plate text from image using Rekognition
    Focus on the biggest, most confident text detections
    """
    try:
        # Call Rekognition to detect text
        response = rekognition.detect_text(
            Image={
                'S3Object': {
                    'Bucket': image_bucket,
                    'Name': image_key
                }
            }
        )
        
        text_detections = response.get('TextDetections', [])
        logger.info(f"Found {len(text_detections)} text detections")
        
        license_plates = []
        
        # Process each text detection
        for detection in text_detections:
            detected_text = detection.get('DetectedText', '')
            confidence = detection.get('Confidence', 0)
            detection_type = detection.get('Type', '')
            
            # Focus on LINE detections (bigger text blocks) and high-confidence WORD detections
            if detection_type in ['LINE', 'WORD']:
                logger.info(f"Checking text: '{detected_text}' (confidence: {confidence:.1f}%, type: {detection_type})")
                
                if is_license_plate_text(detected_text, confidence):
                    # Clean the text for storage
                    clean_text = re.sub(r'[^A-Z0-9\s-]', '', detected_text.upper()).strip()
                    
                    license_plates.append({
                        'text': clean_text,
                        'confidence': confidence,
                        'original': detected_text,
                        'type': detection_type
                    })
                    
                    logger.info(f"License plate found: '{clean_text}' (confidence: {confidence:.1f}%)")
        
        # Sort by confidence (highest first) and remove duplicates
        license_plates.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Remove duplicates based on cleaned text
        unique_plates = []
        seen_texts = set()
        
        for plate in license_plates:
            clean_key = re.sub(r'[\s-]', '', plate['text'])
            if clean_key not in seen_texts:
                seen_texts.add(clean_key)
                unique_plates.append(plate)
        
        return unique_plates
        
    except ClientError as e:
        logger.error(f"Error calling Rekognition: {e}")
        raise

def save_to_dynamodb(license_plate, confidence, image_key, bucket_name):
    """
    Save license plate detection to DynamoDB
    """
    try:
        table = dynamodb.Table(TABLE_NAME)
        
        # Create item
        item = {
            'image_name': image_key,  # Primary key
            'license_plate': license_plate,
            'confidence': Decimal(str(round(confidence, 2))),
            'bucket_name': bucket_name,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Save to DynamoDB
        table.put_item(Item=item)
        logger.info(f"Saved license plate '{license_plate}' to DynamoDB")
        return True
        
    except ClientError as e:
        logger.error(f"Error saving to DynamoDB: {e}")
        return False

def lambda_handler(event, context):
    """
    Simple Lambda handler for license plate detection
    """
    try:
        # Initialize AWS clients
        init_aws_clients()
        
        # Get S3 event details
        record = event['Records'][0]
        bucket_name = record['s3']['bucket']['name']
        image_key = urllib.parse.unquote_plus(record['s3']['object']['key'], encoding='utf-8')
        
        logger.info(f"Processing image: {image_key} from bucket: {bucket_name}")
        
        # Extract license plates from image
        license_plates = extract_license_plates(bucket_name, image_key)
        
        if not license_plates:
            logger.info("No license plates detected")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'No license plates detected',
                    'image_key': image_key
                })
            }
        
        # Process detected license plates
        results = []
        for plate in license_plates:
            # Save to DynamoDB
            saved = save_to_dynamodb(
                plate['text'], 
                plate['confidence'], 
                image_key, 
                bucket_name
            )
            
            results.append({
                'license_plate': plate['text'],
                'confidence': round(plate['confidence'], 2),
                'original_text': plate['original'],
                'detection_type': plate['type'],
                'saved_to_db': saved
            })
        
        logger.info(f"Successfully processed {len(results)} license plates")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f'Found {len(results)} license plates',
                'image_key': image_key,
                'license_plates': results
            })
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'image_key': image_key if 'image_key' in locals() else 'unknown'
            })
        }

# Test function for local testing
def test_with_sample_data():
    """
    Test function with sample Rekognition response
    """
    sample_detections = [
        {'DetectedText': 'CMY6671', 'Confidence': 95.5, 'Type': 'LINE'},
        {'DetectedText': 'PKA CG50', 'Confidence': 92.3, 'Type': 'LINE'},
        {'DetectedText': 'CSY-6171', 'Confidence': 88.7, 'Type': 'LINE'},
        {'DetectedText': 'FLORIDA', 'Confidence': 91.2, 'Type': 'WORD'},
        {'DetectedText': 'STOP', 'Confidence': 85.0, 'Type': 'WORD'},  # Should be filtered out
        {'DetectedText': 'A', 'Confidence': 90.0, 'Type': 'WORD'},     # Too short
    ]
    
    print("Testing license plate detection:")
    for detection in sample_detections:
        text = detection['DetectedText']
        confidence = detection['Confidence']
        is_plate = is_license_plate_text(text, confidence)
        print(f"  '{text}' (confidence: {confidence}%) -> {'✓ License Plate' if is_plate else '✗ Not a license plate'}")

if __name__ == "__main__":
    test_with_sample_data()

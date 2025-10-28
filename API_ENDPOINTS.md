# Pre-Flight Anomaly Detection API

## Public Function URLs

Your Azure Function App is deployed and accessible at:
- **Base URL**: `https://pre-fligt-anomaly-detection.azurewebsites.net`

## Available Endpoints

### 1. Health Check
- **URL**: `https://pre-fligt-anomaly-detection.azurewebsites.net/api/health`
- **Method**: GET
- **Purpose**: Check if the service is running
- **Authentication**: None (Public)

**Example Response**:
```json
{
  "status": "healthy",
  "service": "Pre-Flight Anomaly Detection",
  "version": "1.0.0",
  "timestamp": "2025-10-28T12:00:00.000Z"
}
```

### 2. Anomaly Detection
- **URL**: `https://pre-fligt-anomaly-detection.azurewebsites.net/api/detect_anomalies`
- **Methods**: GET, POST
- **Authentication**: None (Public)

#### GET Request (API Information)
Returns API documentation and sample usage.

**Example**: Open in browser or use curl:
```bash
curl https://pre-fligt-anomaly-detection.azurewebsites.net/api/detect_anomalies
```

#### POST Request (Anomaly Detection)
Submit sensor readings for anomaly analysis.

**Request Body** (JSON):
```json
{
  "altitude": 35000,
  "airspeed": 450,
  "engine_temp": 850,
  "fuel_flow": 2500,
  "hydraulic_pressure": 3000
}
```

**Example Response**:
```json
{
  "anomalies_detected": false,
  "anomalous_readings": [],
  "total_readings": 1
}
```

## Testing the API

### Using curl (Command Line)

1. **Health Check**:
```bash
curl https://pre-fligt-anomaly-detection.azurewebsites.net/api/health
```

2. **Get API Info**:
```bash
curl https://pre-fligt-anomaly-detection.azurewebsites.net/api/detect_anomalies
```

3. **Submit Normal Data**:
```bash
curl -X POST https://pre-fligt-anomaly-detection.azurewebsites.net/api/detect_anomalies \
  -H "Content-Type: application/json" \
  -d '{
    "altitude": 35000,
    "airspeed": 450,
    "engine_temp": 850,
    "fuel_flow": 2500,
    "hydraulic_pressure": 3000
  }'
```

4. **Submit Anomalous Data**:
```bash
curl -X POST https://pre-fligt-anomaly-detection.azurewebsites.net/api/detect_anomalies \
  -H "Content-Type: application/json" \
  -d '{
    "altitude": 35000,
    "airspeed": 450,
    "engine_temp": 1200,
    "fuel_flow": 5000,
    "hydraulic_pressure": 1000
  }'
```

### Using Browser

You can test GET endpoints directly in your browser:
- Health: https://pre-fligt-anomaly-detection.azurewebsites.net/api/health
- API Info: https://pre-fligt-anomaly-detection.azurewebsites.net/api/detect_anomalies

### Using Python

Run the included test script:
```bash
python test_public_api.py
```

## Expected Response Format

### Normal Reading Response
```json
{
  "anomalies_detected": false,
  "anomalous_readings": [],
  "total_readings": 1
}
```

### Anomalous Reading Response
```json
{
  "anomalies_detected": true,
  "anomalous_readings": [
    {
      "altitude": 35000,
      "airspeed": 450,
      "engine_temp": 1200,
      "fuel_flow": 5000,
      "hydraulic_pressure": 1000,
      "anomaly_score": 0.85
    }
  ],
  "total_readings": 1
}
```

## Deployment Status

The function is configured with:
- ✅ **Public Access**: No authentication required
- ✅ **HTTP Methods**: GET and POST supported
- ✅ **CORS**: Enabled for web browser access
- ✅ **Automatic Deployment**: Via GitHub Actions
- ✅ **Model Training**: Randomized data on each deployment

## Monitoring

- Monitor deployments: [GitHub Actions](https://github.com/anthonyricevuto3-lab/Pre-Flight-Anomaly-Detection/actions)
- Function logs: Available in Azure Portal under your Function App
- Live testing: Use the health endpoint to verify uptime
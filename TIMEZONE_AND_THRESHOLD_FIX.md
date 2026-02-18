# Timezone and Threshold Fix - Analysis

## Issues Identified

### 1. Timezone Mismatch Error
**Error in logs:**
```
[ERROR] [utils] [username=dev1@gmail.com] Feature calculation failed: Invalid comparison between dtype=datetime64[us, UTC] and Timestamp
```

**Root Cause:**
- MongoDB stores timestamps as UTC timezone-aware datetimes
- The code was trying to compare timezone-aware datetimes with naive datetimes
- This caused feature calculation to fail and return default features
- Default features made everything look suspicious

**Fix Applied:**
- Added timezone handling in `calculate_features()` and `_extract_features_from_transaction()`
- Convert all timezone-aware datetimes to naive datetimes before comparison
- This allows proper feature calculation from MongoDB data

### 2. Classification Threshold Issue
**Current Thresholds:**
- Fraud: ensemble_fraud_score >= 0.75
- Suspicious: ensemble_fraud_score >= 0.5
- Normal: ensemble_fraud_score < 0.5

**Problem:**
- HMM predicts state 1 (maps to 0.5 probability) for most transactions
- CRF predicts label 1 (maps to 0.5 probability) for most transactions
- Ensemble score = (0.5 + 0.5) / 2 = 0.5
- This triggers "Suspicious" classification for everything

**Why Models Predict 1:**
- The models were trained on synthetic data with specific patterns
- The current feature extraction (when working) produces features that match the "Suspicious" pattern
- Need to investigate if the models are correctly trained or if features are still wrong

## Files Modified

1. **UPI/Backend/app/utils.py**
   - Added timezone-aware datetime handling in `calculate_features()`
   - Added timezone-aware datetime handling in `_extract_features_from_transaction()`
   - Converts UTC datetimes to naive datetimes for comparison

2. **UPI/Backend/app/routes/fraud.py**
   - Added IST timezone handling when parsing transaction_time from API request
   - Assumes IST if no timezone info provided

## Next Steps

1. **Test with timezone fix:**
   - Restart server
   - Send transaction with explicit timezone or IST time
   - Verify feature calculation succeeds (no error logs)
   - Check if predictions improve

2. **Investigate model predictions:**
   - If everything is still suspicious, need to check:
     - Are the models correctly trained?
     - Are the features being calculated correctly?
     - Should the thresholds be adjusted?

3. **Possible solutions:**
   - Retrain models with better synthetic data
   - Adjust classification thresholds
   - Review feature calculation logic

## Expected Behavior After Fix

**With timezone fix:**
- Feature calculation should succeed (no errors)
- Features should be calculated from actual MongoDB data
- Models should make predictions based on real features
- Not everything should be classified as suspicious

**If still suspicious:**
- May need to adjust thresholds
- May need to retrain models
- May need to review feature calculation logic

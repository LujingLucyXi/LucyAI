-- ============================================================================
-- LucyAI Platform - Database Schema & Integrity Metrics Queries
-- Purpose: Code practice for calculating integrity metrics on mock AI platform
-- ============================================================================

-- ============================================================================
-- 1. DATABASE SCHEMA CREATION
-- ============================================================================

-- Users Table: Stores platform users
CREATE TABLE users (
    user_id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    account_status ENUM('active', 'inactive', 'suspended') DEFAULT 'active'
);

-- AI Models Table: Stores AI models created by users
CREATE TABLE models (
    model_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50), -- e.g., 'classification', 'regression', 'nlp'
    creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    version INT DEFAULT 1,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Predictions Table: Stores model predictions/inferences
CREATE TABLE predictions (
    prediction_id INT PRIMARY KEY AUTO_INCREMENT,
    model_id INT NOT NULL,
    input_data JSON,
    predicted_output VARCHAR(255),
    confidence_score DECIMAL(5, 4),
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_verified BOOLEAN DEFAULT FALSE,
    actual_output VARCHAR(255),
    FOREIGN KEY (model_id) REFERENCES models(model_id)
);

-- Model Metrics Table: Stores performance metrics for models
CREATE TABLE model_metrics (
    metric_id INT PRIMARY KEY AUTO_INCREMENT,
    model_id INT NOT NULL,
    accuracy DECIMAL(5, 4),
    precision DECIMAL(5, 4),
    recall DECIMAL(5, 4),
    f1_score DECIMAL(5, 4),
    total_predictions INT,
    calculation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES model_metrics(model_id)
);

-- API Usage Logs Table: Tracks API calls
CREATE TABLE api_logs (
    log_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    model_id INT,
    endpoint VARCHAR(255),
    request_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    response_status INT,
    response_time_ms INT,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (model_id) REFERENCES models(model_id)
);

-- ============================================================================
-- 2. INTEGRITY METRICS QUERIES
-- ============================================================================

-- 2.1 DATA COMPLETENESS METRICS
-- ============================================================================

-- Check for NULL values in critical user fields
SELECT 
    'users' AS table_name,
    COUNT(*) AS total_records,
    SUM(CASE WHEN user_id IS NULL THEN 1 ELSE 0 END) AS null_user_id,
    SUM(CASE WHEN username IS NULL THEN 1 ELSE 0 END) AS null_username,
    SUM(CASE WHEN email IS NULL THEN 1 ELSE 0 END) AS null_email
FROM users;

-- Check for NULL values in predictions
SELECT 
    'predictions' AS table_name,
    COUNT(*) AS total_records,
    SUM(CASE WHEN predicted_output IS NULL THEN 1 ELSE 0 END) AS null_predicted_output,
    SUM(CASE WHEN confidence_score IS NULL THEN 1 ELSE 0 END) AS null_confidence_score
FROM predictions;

-- 2.2 REFERENTIAL INTEGRITY METRICS
-- ============================================================================

-- Find orphaned models (models with non-existent user_id)
SELECT 
    COUNT(*) AS orphaned_models_count
FROM models m
WHERE m.user_id NOT IN (SELECT user_id FROM users);

-- Find orphaned predictions (predictions with non-existent model_id)
SELECT 
    COUNT(*) AS orphaned_predictions_count
FROM predictions p
WHERE p.model_id NOT IN (SELECT model_id FROM models);

-- Find orphaned metrics (metrics with non-existent model_id)
SELECT 
    COUNT(*) AS orphaned_metrics_count
FROM model_metrics mm
WHERE mm.model_id NOT IN (SELECT model_id FROM models);

-- Find orphaned API logs (logs with non-existent user_id or model_id)
SELECT 
    COUNT(*) AS orphaned_logs_count
FROM api_logs al
WHERE al.user_id NOT IN (SELECT user_id FROM users)
   OR (al.model_id IS NOT NULL AND al.model_id NOT IN (SELECT model_id FROM models));

-- 2.3 DUPLICATE DATA DETECTION
-- ============================================================================

-- Find duplicate usernames
SELECT 
    username,
    COUNT(*) AS count
FROM users
GROUP BY username
HAVING COUNT(*) > 1;

-- Find duplicate emails
SELECT 
    email,
    COUNT(*) AS count
FROM users
GROUP BY email
HAVING COUNT(*) > 1;

-- Find duplicate model names per user
SELECT 
    user_id,
    model_name,
    COUNT(*) AS count
FROM models
GROUP BY user_id, model_name
HAVING COUNT(*) > 1;

-- 2.4 DATA CONSISTENCY METRICS
-- ============================================================================

-- Check for confidence scores outside valid range (0-1)
SELECT 
    COUNT(*) AS invalid_confidence_scores
FROM predictions
WHERE confidence_score < 0 OR confidence_score > 1;

-- Check for invalid metric values (should be between 0-1)
SELECT 
    COUNT(*) AS invalid_metrics_count
FROM model_metrics
WHERE accuracy NOT BETWEEN 0 AND 1
   OR precision NOT BETWEEN 0 AND 1
   OR recall NOT BETWEEN 0 AND 1
   OR f1_score NOT BETWEEN 0 AND 1;

-- Check for predictions with future timestamps
SELECT 
    COUNT(*) AS future_predictions
FROM predictions
WHERE prediction_timestamp > NOW();

-- 2.5 MODEL ACTIVITY METRICS
-- ============================================================================

-- Models without any predictions (unused models)
SELECT 
    m.model_id,
    m.model_name,
    u.username,
    m.creation_date
FROM models m
INNER JOIN users u ON m.user_id = u.user_id
WHERE m.model_id NOT IN (SELECT DISTINCT model_id FROM predictions);

-- Models with no metrics calculated
SELECT 
    m.model_id,
    m.model_name,
    COUNT(p.prediction_id) AS prediction_count
FROM models m
LEFT JOIN predictions p ON m.model_id = p.model_id
WHERE m.model_id NOT IN (SELECT DISTINCT model_id FROM model_metrics)
GROUP BY m.model_id, m.model_name;

-- 2.6 USER ACTIVITY METRICS
-- ============================================================================

-- Active users vs inactive users
SELECT 
    account_status,
    COUNT(*) AS user_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM users), 2) AS percentage
FROM users
GROUP BY account_status;

-- Users with no models
SELECT 
    u.user_id,
    u.username,
    u.registration_date
FROM users u
WHERE u.user_id NOT IN (SELECT DISTINCT user_id FROM models);

-- Users with no API activity
SELECT 
    u.user_id,
    u.username,
    u.registration_date
FROM users u
WHERE u.user_id NOT IN (SELECT DISTINCT user_id FROM api_logs WHERE user_id IS NOT NULL);

-- 2.7 PREDICTION QUALITY METRICS
-- ============================================================================

-- Predictions with low confidence
SELECT 
    COUNT(*) AS low_confidence_predictions,
    ROUND(AVG(confidence_score), 4) AS avg_confidence
FROM predictions
WHERE confidence_score < 0.5;

-- Verified predictions accuracy (actual vs predicted)
SELECT 
    model_id,
    COUNT(*) AS verified_count,
    SUM(CASE WHEN predicted_output = actual_output THEN 1 ELSE 0 END) AS correct_predictions,
    ROUND(
        SUM(CASE WHEN predicted_output = actual_output THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 
        2
    ) AS verification_accuracy
FROM predictions
WHERE is_verified = TRUE
GROUP BY model_id;

-- 2.8 COMPREHENSIVE DATA INTEGRITY DASHBOARD
-- ============================================================================

SELECT 
    'Total Users' AS metric, COUNT(*) AS value
FROM users
UNION ALL
SELECT 'Total Models', COUNT(*) FROM models
UNION ALL
SELECT 'Total Predictions', COUNT(*) FROM predictions
UNION ALL
SELECT 'Total API Logs', COUNT(*) FROM api_logs
UNION ALL
SELECT 'Active Users', COUNT(*) FROM users WHERE account_status = 'active'
UNION ALL
SELECT 'Orphaned Models', COUNT(*) FROM models WHERE user_id NOT IN (SELECT user_id FROM users)
UNION ALL
SELECT 'Orphaned Predictions', COUNT(*) FROM predictions WHERE model_id NOT IN (SELECT model_id FROM models)
UNION ALL
SELECT 'Unused Models', COUNT(*) FROM models WHERE model_id NOT IN (SELECT DISTINCT model_id FROM predictions)
UNION ALL
SELECT 'Verified Predictions', COUNT(*) FROM predictions WHERE is_verified = TRUE;

-- 2.9 TIME-SERIES DATA INTEGRITY
-- ============================================================================

-- Check for data consistency: Predictions should be after model creation
SELECT 
    COUNT(*) AS inconsistent_records
FROM predictions p
INNER JOIN models m ON p.model_id = m.model_id
WHERE p.prediction_timestamp < m.creation_date;

-- Check for metrics calculated before first prediction
SELECT 
    COUNT(*) AS inconsistent_metrics
FROM model_metrics mm
WHERE mm.calculation_date < (
    SELECT MIN(prediction_timestamp) 
    FROM predictions p 
    WHERE p.model_id = mm.model_id
);

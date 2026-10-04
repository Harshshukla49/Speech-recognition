"""
Database module for Speech Emotion Recognition & Multimodal Platform.
Provides SQLite storage for analysis history, audio & video metadata retention, filtering, and export.
"""
import os
import sqlite3
import json
from datetime import datetime
import pandas as pd
import config

DB_PATH = os.path.join(config.BASE_DIR, 'data', 'history.db')


def get_connection():
    """Get a database connection"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables and run incremental migrations"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            filename TEXT NOT NULL,
            source_type TEXT NOT NULL,
            duration REAL NOT NULL,
            predicted_emotion TEXT NOT NULL,
            confidence REAL NOT NULL,
            is_low_confidence INTEGER NOT NULL,
            probabilities_json TEXT NOT NULL,
            acoustic_metrics_json TEXT NOT NULL,
            segment_results_json TEXT,
            model_version TEXT NOT NULL,
            processing_time REAL NOT NULL,
            media_type TEXT DEFAULT 'audio',
            sync_offset_ms REAL DEFAULT 0.0,
            sync_quality_score REAL DEFAULT 0.0,
            visual_metrics_json TEXT
        )
    """)
    conn.commit()

    # Dynamic column migrations for existing databases
    cursor.execute("PRAGMA table_info(analyses)")
    existing_columns = [col['name'] for col in cursor.fetchall()]

    new_columns = [
        ('media_type', 'TEXT DEFAULT "audio"'),
        ('sync_offset_ms', 'REAL DEFAULT 0.0'),
        ('sync_quality_score', 'REAL DEFAULT 0.0'),
        ('visual_metrics_json', 'TEXT')
    ]

    for col_name, col_def in new_columns:
        if col_name not in existing_columns:
            try:
                cursor.execute(f"ALTER TABLE analyses ADD COLUMN {col_name} {col_def}")
            except sqlite3.OperationalError:
                pass

    conn.commit()
    conn.close()


def save_analysis(
    analysis_id,
    filename,
    source_type,
    duration,
    predicted_emotion,
    confidence,
    is_low_confidence,
    probabilities,
    acoustic_metrics,
    segment_results=None,
    model_version="CNN-LSTM Hybrid v2.0",
    processing_time=0.0,
    media_type="audio",
    sync_offset_ms=0.0,
    sync_quality_score=0.0,
    visual_metrics=None
):
    """Save an analysis record (audio or video/lip-sync) to the database"""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute("""
        INSERT OR REPLACE INTO analyses (
            id, timestamp, filename, source_type, duration,
            predicted_emotion, confidence, is_low_confidence,
            probabilities_json, acoustic_metrics_json, segment_results_json,
            model_version, processing_time, media_type,
            sync_offset_ms, sync_quality_score, visual_metrics_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        analysis_id,
        timestamp,
        filename,
        source_type,
        float(duration),
        str(predicted_emotion).lower(),
        float(confidence),
        1 if is_low_confidence else 0,
        json.dumps(probabilities),
        json.dumps(acoustic_metrics),
        json.dumps(segment_results) if segment_results else None,
        model_version,
        float(processing_time),
        media_type,
        float(sync_offset_ms or 0.0),
        float(sync_quality_score or 0.0),
        json.dumps(visual_metrics) if visual_metrics else None
    ))
    conn.commit()
    conn.close()


def get_all_analyses(limit=100, offset=0, emotion_filter="All", search_query=""):
    """Retrieve filtered analyses"""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM analyses WHERE 1=1"
    params = []
    
    if emotion_filter and emotion_filter != "All":
        query += " AND predicted_emotion = ?"
        params.append(emotion_filter.lower())
        
    if search_query:
        query += " AND (filename LIKE ? OR id LIKE ?)"
        params.extend([f"%{search_query}%", f"%{search_query}%"])
        
    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    records = []
    for r in rows:
        records.append({
            'id': r['id'],
            'timestamp': r['timestamp'],
            'filename': r['filename'],
            'source_type': r['source_type'],
            'duration': r['duration'],
            'predicted_emotion': r['predicted_emotion'],
            'confidence': r['confidence'],
            'is_low_confidence': bool(r['is_low_confidence']),
            'probabilities': json.loads(r['probabilities_json']),
            'acoustic_metrics': json.loads(r['acoustic_metrics_json']),
            'segment_results': json.loads(r['segment_results_json']) if r['segment_results_json'] else None,
            'model_version': r['model_version'],
            'processing_time': r['processing_time'],
            'media_type': r['media_type'] if 'media_type' in r.keys() else 'audio',
            'sync_offset_ms': r['sync_offset_ms'] if 'sync_offset_ms' in r.keys() else 0.0,
            'sync_quality_score': r['sync_quality_score'] if 'sync_quality_score' in r.keys() else 0.0,
            'visual_metrics': json.loads(r['visual_metrics_json']) if 'visual_metrics_json' in r.keys() and r['visual_metrics_json'] else None
        })
        
    conn.close()
    return records


def get_analysis_by_id(analysis_id):
    """Retrieve a single analysis by ID"""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
        
    return {
        'id': row['id'],
        'timestamp': row['timestamp'],
        'filename': row['filename'],
        'source_type': row['source_type'],
        'duration': row['duration'],
        'predicted_emotion': row['predicted_emotion'],
        'confidence': row['confidence'],
        'is_low_confidence': bool(row['is_low_confidence']),
        'probabilities': json.loads(row['probabilities_json']),
        'acoustic_metrics': json.loads(row['acoustic_metrics_json']),
        'segment_results': json.loads(row['segment_results_json']) if row['segment_results_json'] else None,
        'model_version': row['model_version'],
        'processing_time': row['processing_time'],
        'media_type': row['media_type'] if 'media_type' in row.keys() else 'audio',
        'sync_offset_ms': row['sync_offset_ms'] if 'sync_offset_ms' in row.keys() else 0.0,
        'sync_quality_score': row['sync_quality_score'] if 'sync_quality_score' in row.keys() else 0.0,
        'visual_metrics': json.loads(row['visual_metrics_json']) if 'visual_metrics_json' in row.keys() and row['visual_metrics_json'] else None
    }


def delete_analysis(analysis_id):
    """Delete an analysis by ID"""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
    conn.commit()
    conn.close()


def clear_all_history():
    """Clear all analysis records"""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM analyses")
    conn.commit()
    conn.close()


def get_summary_stats():
    """Calculate summary analytics from actual stored records"""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM analyses")
    total_count = cursor.fetchone()[0]
    
    if total_count == 0:
        conn.close()
        return {
            'total_analyses': 0,
            'avg_confidence': 0.0,
            'avg_processing_time': 0.0,
            'emotion_distribution': {},
            'low_confidence_count': 0,
            'recent_records': []
        }
        
    cursor.execute("SELECT AVG(confidence), AVG(processing_time) FROM analyses")
    avg_conf, avg_proc = cursor.fetchone()
    
    cursor.execute("SELECT COUNT(*) FROM analyses WHERE is_low_confidence = 1")
    low_conf_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT predicted_emotion, COUNT(*) FROM analyses GROUP BY predicted_emotion")
    emotion_dist = dict(cursor.fetchall())
    
    conn.close()
    
    recent_records = get_all_analyses(limit=5)
    
    return {
        'total_analyses': total_count,
        'avg_confidence': float(avg_conf or 0.0),
        'avg_processing_time': float(avg_proc or 0.0),
        'emotion_distribution': emotion_dist,
        'low_confidence_count': low_conf_count,
        'recent_records': recent_records
    }


def export_history_dataframe():
    """Export history to a pandas DataFrame"""
    init_db()
    conn = get_connection()
    df = pd.read_sql_query("SELECT id, timestamp, filename, source_type, media_type, duration, predicted_emotion, confidence, sync_offset_ms, sync_quality_score, is_low_confidence, model_version, processing_time FROM analyses ORDER BY timestamp DESC", conn)
    conn.close()
    return df

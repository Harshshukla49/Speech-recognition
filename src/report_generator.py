"""
PDF Report Generator module for Speech Emotion Recognition & Multimodal Lip-Sync Platform.
Generates professional, presentation-ready PDF analysis summaries using ReportLab.
"""
import io
import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

import config


def generate_pdf_report(
    analysis_record,
    audio_time_series=None,
    sample_rate=22050,
    sync_results=None,
    output_stream=None
):
    """
    Generate a formatted PDF report for an emotion and visual lip-sync analysis record.
    
    Args:
        analysis_record: dict containing analysis data
        audio_time_series: Optional 1D numpy array of audio samples for embedding plots
        sample_rate: Audio sample rate
        sync_results: Optional dict containing lip-sync cross-correlation & tracking metrics
        output_stream: Optional BytesIO or file object. If None, a BytesIO buffer is returned.
        
    Returns:
        BytesIO object containing PDF binary data
    """
    if output_stream is None:
        buffer = io.BytesIO()
    else:
        buffer = output_stream
        
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Brand Styles
    primary_color = colors.HexColor("#0B1120")
    accent_color = colors.HexColor("#0284C7")
    text_dark = colors.HexColor("#1E293B")
    text_muted = colors.HexColor("#64748B")
    
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=primary_color,
        alignment=TA_LEFT
    )
    
    subtitle_style = ParagraphStyle(
        'ReportSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=text_muted,
        alignment=TA_LEFT
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=primary_color,
        spaceBefore=10,
        spaceAfter=5
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=text_dark
    )
    
    badge_style = ParagraphStyle(
        'EmotionBadge',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=accent_color,
        alignment=TA_CENTER
    )
    
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=10.5,
        textColor=text_muted,
        alignment=TA_CENTER
    )
    
    story = []
    
    # -------------------------------------------------------------------------
    # 1. Header Banner
    # -------------------------------------------------------------------------
    is_video = (analysis_record.get('media_type') == 'video' or sync_results is not None or analysis_record.get('visual_metrics') is not None)
    studio_title = "🎙️👁️ MULTIMODAL SPEECH & LIP-SYNC STUDIO" if is_video else "🎙️ SPEECH EMOTION AI STUDIO"
    
    header_data = [
        [
            Paragraph(f"<b>{studio_title}</b>", title_style),
            Paragraph(f"<b>REPORT ID:</b> #{analysis_record.get('id', 'N/A')[:8]}<br/><b>GENERATED:</b> {analysis_record.get('timestamp', 'N/A')}", subtitle_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[4.2 * inch, 3.0 * inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent_color, spaceBefore=3, spaceAfter=10))
    
    # -------------------------------------------------------------------------
    # 2. Executive Summary Hero
    # -------------------------------------------------------------------------
    emotion = analysis_record.get('predicted_emotion', 'neutral').upper()
    conf = analysis_record.get('confidence', 0.0) * 100
    is_low_conf = analysis_record.get('is_low_confidence', False)
    
    summary_data = [
        [
            Paragraph(f"<b>PREDICTED EMOTION</b><br/><font size=18 color='#0284C7'><b>{emotion}</b></font>", badge_style),
            Paragraph(
                f"<b>Media File:</b> {analysis_record.get('filename', 'Unknown')} (Type: {analysis_record.get('media_type', 'audio').upper()})<br/>"
                f"<b>Duration:</b> {analysis_record.get('duration', 0.0)}s | <b>Processing Latency:</b> {analysis_record.get('processing_time', 0.0):.2f}s<br/>"
                f"<b>Classification Confidence:</b> <b>{conf:.2f}%</b> {'(⚠️ Low Confidence)' if is_low_conf else '(High Confidence)'}<br/>"
                f"<b>Inference Pipeline:</b> {analysis_record.get('model_version', 'CNN-LSTM Hybrid v2.0')}",
                body_style
            )
        ]
    ]
    summary_table = Table(summary_data, colWidths=[2.5 * inch, 4.7 * inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#E2E8F0')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 8))
    
    # -------------------------------------------------------------------------
    # 3. Lip-Sync & Visual Articulation Section (If Video Analysis)
    # -------------------------------------------------------------------------
    sync_data = sync_results or analysis_record.get('visual_metrics')
    if sync_data:
        story.append(Paragraph("👁️ Lip-Sync Synchronization & Visual Articulation Analysis", section_heading))
        
        sqi = sync_data.get('sync_quality_index', 0.0)
        offset_ms = sync_data.get('time_offset_ms', 0.0)
        corr = sync_data.get('correlation_coefficient', 0.0)
        status_str = sync_data.get('sync_status', 'Analyzed')
        speaking_act = sync_data.get('speaking_activity_pct', 0.0)
        tracking_con = sync_data.get('tracking_consistency_pct', 0.0)

        sync_table_data = [
            [
                Paragraph(f"<b>Sync Quality Index:</b> {sqi:.1f}%", body_style),
                Paragraph(f"<b>A/V Temporal Offset:</b> {offset_ms:+.1f} ms", body_style),
                Paragraph(f"<b>Cross-Correlation (r):</b> {corr:.3f}", body_style)
            ],
            [
                Paragraph(f"<b>Synchronization State:</b> {status_str}", body_style),
                Paragraph(f"<b>Visual Speech Activity:</b> {speaking_act:.1f}%", body_style),
                Paragraph(f"<b>Face Tracking Consistency:</b> {tracking_con:.1f}%", body_style)
            ]
        ]
        sync_table = Table(sync_table_data, colWidths=[2.4 * inch, 2.4 * inch, 2.4 * inch])
        sync_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F0FDF4') if sqi >= 60 else colors.HexColor('#FFFBEB')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
            ('PADDING', (0, 0), (-1, -1), 5.5),
        ]))
        story.append(sync_table)
        story.append(Spacer(1, 8))

    # -------------------------------------------------------------------------
    # 4. 7-Class Probability Breakdown
    # -------------------------------------------------------------------------
    story.append(Paragraph("7-Class Emotion Probability Distribution", section_heading))
    probs = analysis_record.get('probabilities', {})
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    
    prob_table_data = [["Emotion Category", "Softmax Probability", "Confidence Indicator"]]
    for emo_name, p_val in sorted_probs:
        pct = p_val * 100
        bars = "■" * int(pct / 5)
        prob_table_data.append([
            emo_name.capitalize(),
            f"{pct:.2f}%",
            f"{bars}"
        ])
        
    prob_table = Table(prob_table_data, colWidths=[2.2 * inch, 1.8 * inch, 3.2 * inch])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('PADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 8))
    
    # -------------------------------------------------------------------------
    # 5. Acoustic Signal Characteristics
    # -------------------------------------------------------------------------
    metrics = analysis_record.get('acoustic_metrics', {})
    if metrics:
        story.append(Paragraph("Acoustic Signal Characteristics", section_heading))
        metrics_table_data = [
            [
                Paragraph(f"<b>Mean RMS Energy:</b> {metrics.get('mean_rms', 'N/A')}", body_style),
                Paragraph(f"<b>Zero Crossing Rate:</b> {metrics.get('mean_zcr', 'N/A')}", body_style),
                Paragraph(f"<b>Spectral Centroid:</b> {metrics.get('mean_spectral_centroid_hz', 'N/A')} Hz", body_style)
            ],
            [
                Paragraph(f"<b>Spectral Rolloff:</b> {metrics.get('mean_spectral_rolloff_hz', 'N/A')} Hz", body_style),
                Paragraph(f"<b>Estimated Pitch (F0):</b> {metrics.get('estimated_mean_pitch_hz', 'N/A')} Hz", body_style),
                Paragraph(f"<b>Silence Ratio:</b> {metrics.get('silence_percentage', 'N/A')}%", body_style)
            ]
        ]
        metrics_table = Table(metrics_table_data, colWidths=[2.4 * inch, 2.4 * inch, 2.4 * inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F1F5F9')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 8))
        
    # -------------------------------------------------------------------------
    # 6. Embedded Waveform & Mel-Spectrogram Plots
    # -------------------------------------------------------------------------
    if audio_time_series is not None and len(audio_time_series) > 0:
        story.append(Paragraph("Acoustic Signal Visualizations", section_heading))
        
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 2.0), facecolor='#FAFAFA')
        
        # Plot Waveform
        time_axis = np.linspace(0, len(audio_time_series) / sample_rate, len(audio_time_series))
        axes[0].plot(time_axis, audio_time_series, color='#0284C7', linewidth=0.8)
        axes[0].set_title('Waveform (Time Domain)', fontsize=8.5, fontweight='bold')
        axes[0].set_xlabel('Time (s)', fontsize=7.5)
        axes[0].set_ylabel('Amplitude', fontsize=7.5)
        axes[0].tick_params(labelsize=7)
        axes[0].grid(True, linestyle='--', alpha=0.3)
        
        # Plot Mel-Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_time_series, sr=sample_rate,
            n_mels=getattr(config, 'N_MELS', 128),
            hop_length=getattr(config, 'HOP_LENGTH', 512)
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(
            mel_spec_db, sr=sample_rate, hop_length=getattr(config, 'HOP_LENGTH', 512),
            x_axis='time', y_axis='mel', ax=axes[1], cmap='viridis'
        )
        axes[1].set_title('Mel-Spectrogram (Energy Density)', fontsize=8.5, fontweight='bold')
        axes[1].set_xlabel('Time (s)', fontsize=7.5)
        axes[1].set_ylabel('Freq (Hz)', fontsize=7.5)
        axes[1].tick_params(labelsize=7)
        
        plt.tight_layout()
        
        plot_buf = io.BytesIO()
        plt.savefig(plot_buf, format='png', dpi=180, bbox_inches='tight')
        plt.close(fig)
        plot_buf.seek(0)
        
        story.append(Image(plot_buf, width=7.2 * inch, height=1.7 * inch))
        story.append(Spacer(1, 8))
        
    # -------------------------------------------------------------------------
    # 7. Responsible AI & Limitations Disclaimer
    # -------------------------------------------------------------------------
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor('#E2E8F0'), spaceBefore=2, spaceAfter=6))
    story.append(Paragraph(
        "<b>Scientific Disclaimer & Responsible AI:</b> Speech emotion classification and lip-sync synchronization analysis "
        "estimate computational acoustic cues and facial articulation patterns. Outputs do not constitute psychological assessments, "
        "forensic guarantees, or absolute proof of speaker veracity. A/V temporal offsets can naturally occur from hardware latencies, "
        "Bluetooth codecs, or video muxing container delays.",
        disclaimer_style
    ))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

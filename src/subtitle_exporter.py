"""
Subtitle & Transcript Multi-Format Exporter Module.
Exports speech transcripts and timestamps into SRT, VTT, TXT, CSV, and PDF formats.
"""
import io
import os
import csv
import pandas as pd
from typing import List, Dict
from datetime import timedelta

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT


class SubtitleExporter:
    """Handles export of aligned speech transcripts to industry-standard subtitle and document formats."""

    @staticmethod
    def _format_timestamp_srt(seconds: float) -> str:
        """Converts seconds into SRT timestamp format: HH:MM:SS,mmm"""
        td = timedelta(seconds=max(0.0, seconds))
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _format_timestamp_vtt(seconds: float) -> str:
        """Converts seconds into WebVTT timestamp format: HH:MM:SS.mmm"""
        td = timedelta(seconds=max(0.0, seconds))
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    @classmethod
    def export_to_srt(cls, segments: List[Dict], text_field: str = "hinglish_text") -> str:
        """
        Generates standard SubRip (.srt) subtitle string.
        """
        srt_lines = []
        for idx, seg in enumerate(segments):
            start_str = cls._format_timestamp_srt(seg.get('start_time', 0.0))
            end_str = cls._format_timestamp_srt(seg.get('end_time', 0.0))
            text = seg.get(text_field, seg.get('hinglish_text', ''))
            
            srt_lines.append(f"{idx + 1}")
            srt_lines.append(f"{start_str} --> {end_str}")
            srt_lines.append(f"{text}\n")

        return '\n'.join(srt_lines)

    @classmethod
    def export_to_vtt(cls, segments: List[Dict], text_field: str = "hinglish_text") -> str:
        """
        Generates standard WebVTT (.vtt) subtitle string.
        """
        vtt_lines = ["WEBVTT", ""]
        for idx, seg in enumerate(segments):
            start_str = cls._format_timestamp_vtt(seg.get('start_time', 0.0))
            end_str = cls._format_timestamp_vtt(seg.get('end_time', 0.0))
            text = seg.get(text_field, seg.get('hinglish_text', ''))
            
            vtt_lines.append(f"{idx + 1}")
            vtt_lines.append(f"{start_str} --> {end_str}")
            vtt_lines.append(f"{text}\n")

        return '\n'.join(vtt_lines)

    @classmethod
    def export_to_txt(cls, segments: List[Dict], text_field: str = "hinglish_text") -> str:
        """
        Generates plain text (.txt) transcript with timestamp prefixes.
        """
        txt_lines = []
        for seg in segments:
            t_start = seg.get('start_time', 0.0)
            t_end = seg.get('end_time', 0.0)
            text = seg.get(text_field, seg.get('hinglish_text', ''))
            mod = seg.get('source_modality', 'Transcript')
            txt_lines.append(f"[{t_start:.2f}s - {t_end:.2f}s] ({mod}) {text}")

        return '\n'.join(txt_lines)

    @classmethod
    def export_to_csv(cls, segments: List[Dict]) -> str:
        """
        Generates CSV format string with complete segment metadata.
        """
        output = io.StringIO()
        fieldnames = [
            'segment_id', 'start_time', 'end_time', 'duration',
            'hinglish_text', 'devanagari_text', 'english_translation',
            'confidence', 'source_modality'
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for idx, seg in enumerate(segments):
            dur = round(seg.get('end_time', 0.0) - seg.get('start_time', 0.0), 2)
            writer.writerow({
                'segment_id': idx + 1,
                'start_time': seg.get('start_time', 0.0),
                'end_time': seg.get('end_time', 0.0),
                'duration': dur,
                'hinglish_text': seg.get('hinglish_text', ''),
                'devanagari_text': seg.get('devanagari_text', ''),
                'english_translation': seg.get('english_translation', ''),
                'confidence': seg.get('confidence', 0.0),
                'source_modality': seg.get('source_modality', '')
            })

        return output.getvalue()

    @classmethod
    def export_to_pdf(
        cls,
        segments: List[Dict],
        video_name: str,
        total_duration: float,
        analysis_mode: str = "Audio + Lip Reading",
        detected_emotion: str = None
    ) -> io.BytesIO:
        """
        Generates a publication-grade PDF transcript summary using ReportLab.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        primary_color = colors.HexColor("#0B1120")
        accent_color = colors.HexColor("#0284C7")
        text_dark = colors.HexColor("#1E293B")
        text_muted = colors.HexColor("#64748B")

        title_style = ParagraphStyle(
            'PdfTitle', parent=styles['Normal'],
            fontName='Helvetica-Bold', fontSize=18, leading=22, textColor=primary_color
        )
        sub_style = ParagraphStyle(
            'PdfSub', parent=styles['Normal'],
            fontName='Helvetica', fontSize=9, leading=12, textColor=text_muted
        )
        sec_style = ParagraphStyle(
            'PdfSec', parent=styles['Normal'],
            fontName='Helvetica-Bold', fontSize=11, leading=15, textColor=primary_color,
            spaceBefore=10, spaceAfter=6
        )
        body_style = ParagraphStyle(
            'PdfBody', parent=styles['Normal'],
            fontName='Helvetica', fontSize=8.5, leading=11.5, textColor=text_dark
        )

        story = []

        # Header
        header_table = Table([
            [
                Paragraph("<b>👄 AI LIP-READING & HINGLISH TRANSCRIPT</b>", title_style),
                Paragraph(f"<b>FILE:</b> {video_name}<br/><b>MODE:</b> {analysis_mode}", sub_style)
            ]
        ], colWidths=[4.6 * inch, 2.6 * inch])
        header_table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
        story.append(header_table)
        story.append(HRFlowable(width="100%", thickness=1.5, color=accent_color, spaceBefore=4, spaceAfter=10))

        # Metadata Card
        meta_info = f"<b>Duration:</b> {total_duration:.1f}s | <b>Total Clauses:</b> {len(segments)}"
        if detected_emotion:
            meta_info += f" | <b>Vocal Affect:</b> {detected_emotion.upper()}"

        meta_table = Table([
            [Paragraph(f"<b>Summary:</b> {meta_info}<br/><b>Default Representation:</b> Natural Conversational Hinglish (Roman Script)", body_style)]
        ], colWidths=[7.2 * inch])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#E2E8F0')),
            ('PADDING', (0, 0), (-1, -1), 8)
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 10))

        # Transcript Table
        story.append(Paragraph("Segment-by-Segment Hinglish Transcript", sec_style))
        table_data = [["Time (s)", "Hinglish Transcript", "English Translation", "Modality", "Conf"]]

        for idx, seg in enumerate(segments[:25]):  # Up to 25 segments per report page
            t_range = f"{seg.get('start_time', 0.0):.1f}s - {seg.get('end_time', 0.0):.1f}s"
            hinglish = seg.get('hinglish_text', '')
            english = seg.get('english_translation', '')
            mod = seg.get('source_modality', 'A+V')
            conf_str = f"{seg.get('confidence', 0.0)*100:.0f}%"

            table_data.append([
                Paragraph(t_range, body_style),
                Paragraph(f"<b>{hinglish}</b>", body_style),
                Paragraph(english, body_style),
                Paragraph(mod, body_style),
                Paragraph(conf_str, body_style)
            ])

        trans_table = Table(table_data, colWidths=[1.1 * inch, 2.5 * inch, 2.2 * inch, 0.9 * inch, 0.5 * inch])
        trans_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), primary_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(trans_table)

        # Disclaimer
        story.append(Spacer(1, 12))
        story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor('#E2E8F0'), spaceBefore=2, spaceAfter=6))
        story.append(Paragraph(
            "<b>Scientific Notice:</b> Visual speech recognition estimates phonemes from visible lip kinematics. "
            "Homophenes (words with identical lip movements, e.g. /p, b, m/) are resolved using phonetic lexicons and optional acoustic signals.",
            ParagraphStyle('Discl', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=7.5, leading=10, textColor=text_muted, alignment=TA_CENTER)
        ))

        doc.build(story)
        buffer.seek(0)
        return buffer

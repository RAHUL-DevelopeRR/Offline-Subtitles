"""
Tests for the SRT Writer module.
"""

import pytest
from pathlib import Path
from pipeline.srt_writer import SRTWriter
from pipeline.merger import SubtitleEntry


@pytest.fixture
def writer():
    return SRTWriter()


@pytest.fixture
def sample_entries():
    return [
        SubtitleEntry(1, 1.2, 4.8, "Hello everyone, welcome to the show."),
        SubtitleEntry(2, 5.1, 6.3, "(Audience clapping)"),
        SubtitleEntry(3, 6.5, 10.2, "Today we're going to talk about something amazing."),
        SubtitleEntry(4, 10.5, 11.8, "(Door slams)"),
    ]


class TestTimestampFormat:
    """Test SRT timestamp formatting."""

    def test_zero(self, writer):
        assert writer._format_timestamp(0.0) == "00:00:00,000"

    def test_simple_seconds(self, writer):
        assert writer._format_timestamp(5.0) == "00:00:05,000"

    def test_milliseconds(self, writer):
        assert writer._format_timestamp(1.234) == "00:00:01,234"

    def test_minutes(self, writer):
        assert writer._format_timestamp(65.5) == "00:01:05,500"

    def test_hours(self, writer):
        assert writer._format_timestamp(3661.123) == "01:01:01,123"

    def test_large_value(self, writer):
        assert writer._format_timestamp(7200.0) == "02:00:00,000"

    def test_negative_clamps_to_zero(self, writer):
        assert writer._format_timestamp(-1.0) == "00:00:00,000"

    def test_fractional_milliseconds(self, writer):
        # 1.5556 → should round to 556ms
        result = writer._format_timestamp(1.5556)
        assert result == "00:00:01,556"


class TestSRTWrite:
    """Test SRT file writing."""

    def test_write_creates_file(self, writer, sample_entries, tmp_path):
        output = tmp_path / "test.srt"
        writer.write(sample_entries, output)
        assert output.exists()

    def test_write_utf8_encoding(self, writer, tmp_path):
        entries = [SubtitleEntry(1, 0.0, 1.0, "Héllo wörld — ñ")]
        output = tmp_path / "utf8.srt"
        writer.write(entries, output)
        content = output.read_text(encoding="utf-8")
        assert "Héllo wörld — ñ" in content

    def test_write_sequential_indices(self, writer, sample_entries, tmp_path):
        output = tmp_path / "indexed.srt"
        writer.write(sample_entries, output)
        content = output.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        # First line of each block should be the index
        indices = [l for l in lines if l.strip().isdigit()]
        assert indices == ["1", "2", "3", "4"]

    def test_write_correct_format(self, writer, tmp_path):
        entries = [SubtitleEntry(1, 1.2, 4.8, "Hello world.")]
        output = tmp_path / "format.srt"
        writer.write(entries, output)
        content = output.read_text(encoding="utf-8")
        expected = "1\n00:00:01,200 --> 00:00:04,800\nHello world.\n\n"
        assert content == expected

    def test_write_empty_entries(self, writer, tmp_path):
        output = tmp_path / "empty.srt"
        writer.write([], output)
        assert output.exists()
        assert output.read_text() == ""

    def test_write_creates_parent_dirs(self, writer, sample_entries, tmp_path):
        output = tmp_path / "sub" / "dir" / "test.srt"
        writer.write(sample_entries, output)
        assert output.exists()


class TestPreview:
    """Test the preview formatter."""

    def test_preview_limits_entries(self, writer, sample_entries):
        preview = writer.write_preview(sample_entries, max_entries=2)
        lines = preview.strip().split("\n")
        assert len(lines) == 3  # 2 entries + "and X more"
        assert "2 more" in lines[-1]

    def test_preview_truncates_long_text(self, writer):
        long_text = "A" * 100
        entries = [SubtitleEntry(1, 0.0, 1.0, long_text)]
        preview = writer.write_preview(entries)
        assert "..." in preview

    def test_preview_empty(self, writer):
        preview = writer.write_preview([])
        assert preview == ""

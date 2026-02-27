"""
Tests for the Timeline Merger module.
"""

import pytest
from pipeline.merger import TimelineMerger, SubtitleEntry
from pipeline.asr_worker import TranscriptionResult
from pipeline.sed_worker import SoundEventResult
from config import MergeConfig


@pytest.fixture
def merger():
    config = MergeConfig(min_duration=0.8, max_duration=7.0, merge_gap=0.5)
    return TimelineMerger(config)


class TestBasicMerge:
    """Test basic merging of transcriptions and sound events."""

    def test_speech_only(self, merger):
        transcriptions = [
            TranscriptionResult(1.0, 4.0, "Hello world", -0.5, "en"),
            TranscriptionResult(5.0, 8.0, "How are you", -0.4, "en"),
        ]
        entries = merger.merge(transcriptions, [])
        assert len(entries) >= 2
        assert entries[0].text == "Hello world"

    def test_sound_only(self, merger):
        sounds = [
            SoundEventResult(1.0, 2.0, "Door", 0.8, "(Door)"),
            SoundEventResult(5.0, 6.0, "Engine", 0.7, "(Engine running)"),
        ]
        entries = merger.merge([], sounds)
        assert len(entries) == 2
        assert "(Door)" in entries[0].text

    def test_empty_inputs(self, merger):
        entries = merger.merge([], [])
        assert len(entries) == 0

    def test_sorted_by_time(self, merger):
        transcriptions = [
            TranscriptionResult(5.0, 8.0, "Second", -0.4, "en"),
            TranscriptionResult(1.0, 4.0, "First", -0.5, "en"),
        ]
        entries = merger.merge(transcriptions, [])
        assert entries[0].start_sec < entries[1].start_sec


class TestOverlapResolution:
    """Test overlap handling between speech and sound events."""

    def test_sound_during_speech_appended(self, merger):
        transcriptions = [
            TranscriptionResult(1.0, 5.0, "Hello everyone", -0.5, "en"),
        ]
        sounds = [
            SoundEventResult(2.0, 3.0, "Clapping", 0.8, "(Clapping)"),
        ]
        entries = merger.merge(transcriptions, sounds)
        # The sound should be appended to the speech entry
        combined = " ".join(e.text for e in entries)
        assert "Hello everyone" in combined
        assert "(Clapping)" in combined

    def test_sound_in_gap_standalone(self, merger):
        transcriptions = [
            TranscriptionResult(1.0, 3.0, "Hello", -0.5, "en"),
            TranscriptionResult(8.0, 10.0, "World", -0.4, "en"),
        ]
        sounds = [
            SoundEventResult(4.0, 6.0, "Door", 0.8, "(Door slams)"),
        ]
        entries = merger.merge(transcriptions, sounds)
        # Door should be a separate entry
        assert any("(Door slams)" == e.text for e in entries)


class TestDurationLimits:
    """Test min/max subtitle duration enforcement."""

    def test_min_duration_enforced(self, merger):
        transcriptions = [
            TranscriptionResult(1.0, 1.2, "Hi", -0.5, "en"),  # Only 0.2s
        ]
        entries = merger.merge(transcriptions, [])
        assert entries[0].duration >= merger.min_duration

    def test_long_entry_split(self, merger):
        long_text = " ".join(["word"] * 50)
        transcriptions = [
            TranscriptionResult(0.0, 15.0, long_text, -0.5, "en"),  # 15s > 7s max
        ]
        entries = merger.merge(transcriptions, [])
        assert len(entries) >= 2  # Should be split
        for entry in entries:
            assert entry.duration <= merger.max_duration + 0.5  # Small tolerance


class TestSequentialIndices:
    """Test that output entries have correct sequential indices."""

    def test_indices_sequential(self, merger):
        transcriptions = [
            TranscriptionResult(1.0, 3.0, "One", -0.5, "en"),
            TranscriptionResult(4.0, 6.0, "Two", -0.5, "en"),
            TranscriptionResult(7.0, 9.0, "Three", -0.5, "en"),
        ]
        entries = merger.merge(transcriptions, [])
        for i, entry in enumerate(entries):
            assert entry.index == i + 1

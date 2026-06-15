import threading

import pytest

from tokencircuit.ring_buffer import RingBuffer


def make_entry(state_hash="a", tool_sig="tool()", iteration=1):
    return {
        "state_hash": state_hash,
        "tool_type_signature": tool_sig,
        "iteration": iteration,
    }


class TestRingBuffer:
    def test_push_and_window(self):
        buf = RingBuffer(maxlen=3)
        buf.push(make_entry(iteration=1))
        buf.push(make_entry(iteration=2))
        assert len(buf.window()) == 2
        assert buf.window()[0]["iteration"] == 1
        assert buf.window()[1]["iteration"] == 2

    def test_is_full(self):
        buf = RingBuffer(maxlen=3)
        assert not buf.is_full()
        buf.push(make_entry(iteration=1))
        assert not buf.is_full()
        buf.push(make_entry(iteration=2))
        assert not buf.is_full()
        buf.push(make_entry(iteration=3))
        assert buf.is_full()

    def test_maxlen_enforcement(self):
        buf = RingBuffer(maxlen=2)
        buf.push(make_entry(iteration=1))
        buf.push(make_entry(iteration=2))
        buf.push(make_entry(iteration=3))
        assert buf.is_full()
        assert len(buf.window()) == 2
        assert buf.window()[0]["iteration"] == 2
        assert buf.window()[1]["iteration"] == 3

    def test_reset(self):
        buf = RingBuffer(maxlen=3)
        buf.push(make_entry(iteration=1))
        buf.push(make_entry(iteration=2))
        buf.reset()
        assert len(buf.window()) == 0
        assert not buf.is_full()

    def test_invalid_maxlen(self):
        with pytest.raises(ValueError, match="maxlen must be >= 1"):
            RingBuffer(maxlen=0)

    def test_missing_keys_raises(self):
        buf = RingBuffer(maxlen=3)
        with pytest.raises(KeyError):
            buf.push({"state_hash": "a"})

    def test_thread_safety(self):
        buf = RingBuffer(maxlen=100)
        errors = []

        def pusher(start, count):
            try:
                for i in range(start, start + count):
                    buf.push(make_entry(iteration=i))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=pusher, args=(0, 50)),
            threading.Thread(target=pusher, args=(50, 50)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(buf.window()) == 100

    def test_window_is_copy(self):
        buf = RingBuffer(maxlen=3)
        buf.push(make_entry(iteration=1))
        w = buf.window()
        w.append(make_entry(iteration=2))
        assert len(buf.window()) == 1

    def test_repr(self):
        buf = RingBuffer(maxlen=3)
        buf.push(make_entry(iteration=1))
        r = repr(buf)
        assert "RingBuffer" in r
        assert "maxlen=3" in r

    def test_maxlen_property(self):
        buf = RingBuffer(maxlen=7)
        assert buf.maxlen == 7

    def test_push_preserves_entry_schema(self):
        buf = RingBuffer(maxlen=3)
        buf.push(make_entry(state_hash="abc", tool_sig="search(str)", iteration=5))
        entry = buf.window()[0]
        assert entry["state_hash"] == "abc"
        assert entry["tool_type_signature"] == "search(str)"
        assert entry["iteration"] == 5

    def test_repeated_push_overwrites_oldest(self):
        buf = RingBuffer(maxlen=3)
        for i in range(5):
            buf.push(make_entry(iteration=i))
        assert len(buf.window()) == 3
        assert [e["iteration"] for e in buf.window()] == [2, 3, 4]

    def test_empty_buffer_len(self):
        buf = RingBuffer(maxlen=3)
        assert len(buf) == 0

    def test_full_buffer_len(self):
        buf = RingBuffer(maxlen=3)
        for i in range(3):
            buf.push(make_entry(iteration=i))
        assert len(buf) == 3

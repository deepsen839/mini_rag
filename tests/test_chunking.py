from app.chunking import chunk_text

def test_chunking_empty():
    assert chunk_text('') == []

def test_chunking_preserves_content_and_overlap():
    words = [f'w{i}' for i in range(210)]
    chunks = chunk_text(' '.join(words), size=100, overlap=20)
    assert len(chunks) == 3
    first, second = chunks[0].split(), chunks[1].split()
    assert first[-20:] == second[:20]
    assert chunks[0].split()[0] == 'w0'
    assert chunks[-1].split()[-1] == 'w209'

def chunker(arrays, chunk_size):

    chunks = []

    for pos in range(0, len(arrays[0]), chunk_size):

        chunk = []

        for array in arrays:

            chunk.append(array[pos: pos+chunk_size])

        chunks.append(chunk)

    return chunks

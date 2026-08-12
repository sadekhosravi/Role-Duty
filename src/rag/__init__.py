"""Configuration, embeddings, and the Chroma client the other packages share.

This package used to be the whole non-graph pipeline — parse, chunk, embed,
search. The parse moved to graph_rag.extraction when the two ingests drifted,
and the chunking is gone entirely: src/doctree keeps documents in sections now,
and owns what goes into Chroma. What remains here is what every path needs and
nobody owns — settings, the embedding function, and opening a collection.
"""

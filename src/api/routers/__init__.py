"""One module per resource, each owning its own routes and nothing else.

    health.py     is the server up, and what does it have to work with
    documents.py  the PDFs: upload, list, remove from the stores
    ingest.py     the two ingests, submitted as background jobs
    jobs.py       what those jobs are doing
    query.py      the three retrieval paths: vector, graph, BM25
    agent.py      the workflow, as a conversation
    tickets.py    what the agent filed

main.py includes them and decides nothing else about them, so a new resource is
a new module and one `include_router` line.
"""

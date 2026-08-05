"""The HTTP layer: the same pipelines the scripts/ CLIs drive, over FastAPI.

    main.py       the app itself — lifespan, error handling, OpenAPI metadata
    settings.py   what the server reads from the environment
    stores.py     the process-wide store handles, and the graph write lock
    paths.py      turning a path a client sent into one it is allowed to touch
    jobs.py       background jobs, for the work too slow to answer in a request
    sessions.py   the agent's conversations, kept per session id
    schemas.py    every request and response body
    routers/      one module per resource

Nothing here implements retrieval, ingestion or the agent. Each router calls the
same function the matching CLI calls — `rag.pipeline.ingest_pdf`,
`graph_rag.query`, `agent.graph.run_workflow` — so the HTTP surface and the
command line cannot answer the same question differently.

The one thing this layer does own is the difference between a process that runs
once and a process that stays up. A CLI opens a store, uses it and exits; a
server has to share one set of handles across concurrent requests, keep a
conversation alive between them, and not block its own event loop while Docling
spends a minute on a PDF. That is what stores.py, sessions.py and jobs.py are.
"""

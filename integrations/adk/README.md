# foveate-adk

[Foveate](https://github.com/sachncs/foveate) for Google ADK. Foveate itself depends on no agent
framework; this small package is the adapter.

```bash
pip install foveate-adk
```

See the docstring of `foveate_adk` for the two entry points: a history compressor for the
framework's message type and `document_tools`, which gives the agent `read_pages`,
`search_document` and `document_outline`.

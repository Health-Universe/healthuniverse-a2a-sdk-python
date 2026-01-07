# Documents API

## DocumentClient

The `DocumentClient` provides methods for reading and writing documents in a Health Universe thread.

Access via `context.document_client` in your agent's `process_message` method.

```python
async def process_message(self, message: str, context: AgentContext) -> str:
    # List all documents
    docs = await context.document_client.list_documents()

    # Filter by name
    protocols = await context.document_client.filter_by_name("protocol")

    # Download content
    content = await context.document_client.download_text(doc.id)

    # Write new document
    await context.document_client.write(
        name="Results",
        content=json.dumps(data),
        filename="results.json"
    )
```

::: health_universe_a2a.DocumentClient

## Document

Metadata object representing a document in a thread.

::: health_universe_a2a.Document

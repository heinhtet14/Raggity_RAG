# Using Raggity from a companion agent

There are two ways to plug it in. Pick based on who should write the reply.

| | `/search` as a tool | `/answer` |
|---|---|---|
| Who writes the reply | your agent, in its own persona and voice | Raggity's LLM |
| Conversation history | your agent has it | not seen |
| Model calls per turn | one (yours) | two (yours + Raggity's) |
| Use when | almost always | a quick standalone Q&A, or testing retrieval quality |

For a companion agent, give it **`/search` as a tool**. The agent decides when to look
something up, reads the passages, and answers in character with the conversation in mind.

## Tool definition

Any tool-calling model takes a JSON-schema tool like this one. Keep the description
specific, because the model decides when to call the tool based on it.

```json
{
  "name": "search_knowledge",
  "description": "Search the user's uploaded documents and notes. Use it when the user asks about something that may be in their files, or when you need a fact you don't already know from the conversation. Returns passages with a filename and page. Cite them when you use them.",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "A standalone search query. Resolve pronouns from the conversation, e.g. 'Lisbon trip hotel booking' rather than 'that booking'."
      },
      "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5}
    },
    "required": ["query"],
    "additionalProperties": false
  }
}
```

Leave `mode`, `rerank` and `filter` out of the tool and set them in your handler. The
model shouldn't choose whose documents it can see.

## Handler

```python
import httpx

RAGGITY = httpx.Client(base_url="http://localhost:8000", timeout=60)


def search_knowledge(query: str, user_id: str, top_k: int = 5) -> str:
    response = RAGGITY.post(
        "/v1/collections/companion/search",
        json={
            "query": query,
            "top_k": top_k,
            "mode": "hybrid",
            "filter": {"user_id": user_id},  # set by your code, never by the model
        },
    )
    response.raise_for_status()
    results = response.json()["results"]
    if not results:
        return "No matching passages."
    return "\n\n".join(
        f"[{r['filename']}{', p.' + str(r['page']) if r['page'] else ''}]\n{r['text']}"
        for r in results
    )
```

## Feeding it memories

Things the agent learns in conversation can go in as text documents too, tagged with the
user:

```bash
curl -X POST localhost:8000/v1/collections/companion/documents/text \
  -H 'content-type: application/json' \
  -d '{"filename": "note-2026-10-22", "text": "Hein prefers morning reminders.", "metadata": {"user_id": "u1", "kind": "memory"}}'
```

Then the handler can search only memories (`"filter": {"user_id": "u1", "kind": "memory"}`)
or only uploaded files.

This works for simple facts, but it is a knowledge base, not episodic memory. It doesn't
merge duplicates, it doesn't decay old facts, and "what did we talk about last week"
isn't a similarity search. A separate memory layer is a good follow-up project.

## Security notes before exposing it

The service has **no authentication** and trusts the `filter` it is given, so run it only
on a private network next to your agent. Before exposing it, add an API key or JWT check, and
derive `user_id` from the caller's identity rather than from the request body.

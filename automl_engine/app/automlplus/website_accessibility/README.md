## Usage example
- Start the unified engine from the repo root
```
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

Analyze an uploaded HTML file:

```
curl -X POST http://localhost:8001/automl/automl_plus/web_access/analyze/ \
  -H "Content-Type: multipart/form-data" \
  -F "file=@./sample_data/test.html"
```

Or analyze a whole website by URL — the service recursively crawls same-origin
links up to `depth` levels (default 2) and analyses every page it fetches:

```
curl -X POST http://localhost:8001/automl/automl_plus/web_access/analyze/ \
  -H "Content-Type: multipart/form-data" \
  -F "url=https://alfie-project.eu" \
  -F "depth=2"
```

The JSON response includes the per-chunk WCAG findings (`results`, tagged with
the page each chunk came from), the combined `readability` metrics, and a
`summary` field produced by an LLM call that aggregates recurring issues,
scores, and readability into a markdown report.

By default the response also carries a rendered, human-readable HTML report of
the same results in the `html_report` field (disable with `-F include_html=false`):

```
jq -r .html_report report.json > report.html && open report.html
```

Existing JSON reports can be rendered offline with the standalone script:

```
uv run python app/automlplus/render_accessibility_report.py report.json report.html
```

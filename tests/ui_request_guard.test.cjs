// Run with: node --test tests/ui_request_guard.test.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

function page(response) {
  const calls = [];
  const nodes = [];
  let reloads = 0;
  const window = {
    location: { href: 'http://localhost:7862/index/?__theme=dark', origin: 'http://localhost:7862', reload: () => reloads++ },
    fetch: async (...args) => { calls.push(args); return response(); }
  };
  const document = {
    body: { appendChild: node => nodes.push(node) },
    getElementById: id => nodes.find(node => node.id === id),
    createElement: tag => ({ tag, style: {}, setAttribute() {}, append(...children) { this.children = children; } })
  };
  const script = fs.readFileSync(path.join(__dirname, '../ui_assets/request_guard.js'), 'utf8')
    .replace('__INDEXTTS_INSTANCE_ID__', JSON.stringify('current-launch'));
  vm.runInNewContext(script, { window, document, URL, Request, Response, Headers });
  return { window, calls, nodes, reloads: () => reloads };
}

test('stamps event requests while preserving bodies, headers, and request options', async () => {
  const p = page(() => new Response('{}', { status: 200 }));
  const req = new Request('http://localhost:7862/index/gradio_api/queue/join', {
    method: 'POST', body: '{"data":["text"]}', headers: { 'Content-Type': 'application/json', 'X-Custom': 'kept' }
  });
  await p.window.fetch(req, { credentials: 'include' });
  const [input, options] = p.calls[0];
  assert.equal(input, req);
  assert.equal(await input.clone().text(), '{"data":["text"]}');
  assert.equal(options.credentials, 'include');
  assert.equal(options.headers.get('X-Custom'), 'kept');
  assert.equal(options.headers.get('x-indextts-ui-instance'), 'current-launch');
  await p.window.fetch('/gradio_api/api/predict', { method: 'POST', body: '{}', headers: { 'X-Second': 'kept' } });
  assert.equal(p.calls[1][1].body, '{}');
  assert.equal(p.calls[1][1].headers.get('X-Second'), 'kept');
});

test('does not change assets, uploads, streams, or other origins', async () => {
  const p = page(() => new Response('{}', { status: 200 }));
  for (const [url, init] of [
    ['/config', {}], ['/gradio_api/queue/data?session_hash=test', {}],
    ['/gradio_api/upload', { method: 'POST', body: 'file' }],
    ['http://localhost:7863/gradio_api/queue/join', { method: 'POST', body: '{}' }]
  ]) {
    await p.window.fetch(url, init);
    assert.equal(p.calls.at(-1)[1], init);
  }
});

test('shows a single reload notice and stops repeated network requests after a stale response', async () => {
  const p = page(() => new Response(JSON.stringify({ code: 'stale_ui', error: 'Copy unsaved text, then reload.' }), { status: 409 }));
  const response = await p.window.fetch('/gradio_api/run/predict', { method: 'POST', body: '{}' });
  assert.equal((await response.json()).code, 'stale_ui');
  assert.equal(p.nodes.length, 1);
  assert.equal(p.nodes[0].id, 'indextts-stale-session');
  assert.equal(p.nodes[0].children[0].textContent, 'Copy unsaved text, then reload.');
  const repeated = await p.window.fetch('/gradio_api/queue/join', { method: 'POST', body: '{}' });
  assert.equal(repeated.status, 409);
  assert.equal(p.calls.length, 1);
  assert.equal(p.reloads(), 0);
  p.nodes[0].children[1].onclick();
  assert.equal(p.reloads(), 1);
});

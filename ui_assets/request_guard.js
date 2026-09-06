// Instance IDs stay in this page's closure so another tab cannot update them.
(() => {
  const instance = __INDEXTTS_INSTANCE_ID__;
  const originalFetch = window.fetch.bind(window);
  const eventPath = /\/gradio_api\/(?:(?:run|api|call)\/(?:v2\/)?[^/]+|queue\/join|cancel)\/?$/;
  let staleMessage = null;

  function showReloadNotice(message) {
    if (document.getElementById("indextts-stale-session")) return;
    const notice = document.createElement("div");
    notice.id = "indextts-stale-session";
    notice.setAttribute("role", "alert");
    notice.style.cssText = "position:fixed;top:0;left:0;right:0;z-index:10000;padding:16px;background:#312512;color:#fff;border-bottom:2px solid #f4b942;font:16px system-ui;display:flex;align-items:center;gap:16px;flex-wrap:wrap";
    const text = document.createElement("span");
    text.textContent = message;
    const reload = document.createElement("button");
    reload.textContent = "Reload IndexTTS";
    reload.style.cssText = "padding:8px 16px;background:#f4b942;color:#111;border:0;border-radius:6px;cursor:pointer";
    reload.onclick = () => window.location.reload();
    notice.append(text, reload);
    document.body.appendChild(notice);
  }

  window.fetch = async (input, init) => {
    const url = new URL(input instanceof Request ? input.url : input, window.location.href);
    const method = (init?.method || (input instanceof Request ? input.method : "GET")).toUpperCase();
    if (url.origin !== window.location.origin || method !== "POST" || !eventPath.test(url.pathname)) {
      return originalFetch(input, init);
    }
    if (staleMessage) {
      return new Response(JSON.stringify({ error: staleMessage, detail: staleMessage, code: "stale_ui" }), {
        status: 409, headers: { "Content-Type": "application/json" }
      });
    }
    const headers = new Headers(init?.headers || (input instanceof Request ? input.headers : undefined));
    headers.set("x-indextts-ui-instance", instance);
    const response = await originalFetch(input, { ...init, headers });
    if (response.status === 409) {
      const body = await response.clone().json().catch(() => null);
      if (body?.code === "stale_ui") {
        staleMessage = body.error;
        if (document.body) showReloadNotice(staleMessage);
        else document.addEventListener("DOMContentLoaded", () => showReloadNotice(staleMessage), { once: true });
      }
    }
    return response;
  };
})();

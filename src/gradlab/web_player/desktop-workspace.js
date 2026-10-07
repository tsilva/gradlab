// Native webviews run in separate processes; BroadcastChannel is not portable between them.
export async function encodePeer(message) {
  if (!(message.blob instanceof Blob)) return JSON.stringify(message);
  const bytes = new Uint8Array(await message.blob.arrayBuffer());
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 8192) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 8192));
  }
  return JSON.stringify({ ...message, blob: { base64: btoa(binary), type: message.blob.type } });
}

export function decodePeer(text) {
  const message = JSON.parse(text);
  if (message.blob?.base64 !== undefined) {
    message.blob = new Blob([Uint8Array.from(atob(message.blob.base64), c => c.charCodeAt(0))],
      { type: message.blob.type });
  }
  return message;
}

export function desktopChannel(token) {
  const channel = new EventTarget();
  const url = new URL("/api/desktop/peer", location.href);
  url.protocol = "ws:";
  const socket = new WebSocket(url);
  let ready = false;
  let pending = Promise.resolve();
  const queue = [];
  socket.addEventListener("open", () => socket.send(JSON.stringify({ token })));
  socket.addEventListener("message", event => {
    const message = decodePeer(event.data);
    if (message.ready) {
      ready = true;
      for (const text of queue.splice(0)) socket.send(text);
    } else channel.dispatchEvent(new MessageEvent("message", { data: message }));
  });
  channel.postMessage = message => {
    // Preserve cursor/frame ordering while Blob conversion is asynchronous.
    pending = pending.then(async () => {
      const text = await encodePeer(message);
      if (ready && socket.readyState === WebSocket.OPEN) socket.send(text);
      else if (socket.readyState <= WebSocket.OPEN && queue.length < 64) queue.push(text);
    }).catch(error => console.error("Desktop workspace synchronization failed", error));
  };
  window.addEventListener("pagehide", () => socket.close(), { once: true });
  return channel;
}

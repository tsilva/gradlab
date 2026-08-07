const message = { type: "gradlab-youtube-oauth-complete" };
const fragment = new URLSearchParams(location.hash.slice(1));
const token = fragment.get("token") || "";

document.querySelector("#oauth-complete-close").addEventListener("click", () => window.close());

if (window.opener && window.opener !== window) {
  window.opener.postMessage(message, location.origin);
  window.close();
} else if (token) {
  location.replace(`/#token=${encodeURIComponent(token)}`);
} else {
  document.querySelector("#oauth-complete-status").textContent = (
    "Authorization succeeded. Close this window and return to the player."
  );
}

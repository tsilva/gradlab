// A real exit action for browser drivers whose Escape key stays inside the page.
// This is served only by the synthetic loopback fixture, never by the player.
document.addEventListener("fullscreenchange", () => {
  document.querySelector("[data-fixture-exit-fullscreen]")?.remove();
  if (!document.fullscreenElement) return;
  const button = document.createElement("button");
  button.dataset.fixtureExitFullscreen = "";
  button.textContent = "Exit fullscreen test";
  button.style.cssText = "position:absolute;top:50px;right:20px;z-index:100";
  button.addEventListener("click", () => document.exitFullscreen());
  document.fullscreenElement.append(button);
});

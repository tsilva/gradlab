function labelFor(button) {
  return (
    button.getAttribute("title") ||
    button.getAttribute("aria-label") ||
    button.textContent?.trim().replace(/\s+/g, " ") ||
    ""
  );
}

export function installButtonTooltips(root = document) {
  const tooltip = root.createElement("div");
  tooltip.className = "button-tooltip";
  tooltip.setAttribute("role", "tooltip");
  tooltip.setAttribute("popover", "manual");
  root.body.append(tooltip);

  let active = null;
  let savedTitle = null;

  function hide() {
    if (active && savedTitle !== null && !active.hasAttribute("title")) {
      active.setAttribute("title", savedTitle);
    }
    active = null;
    savedTitle = null;
    if (tooltip.matches(":popover-open")) tooltip.hidePopover();
  }

  function show(button) {
    if (button === active) return;
    hide();
    const label = labelFor(button);
    if (!label) return;
    active = button;
    savedTitle = button.getAttribute("title");
    if (savedTitle !== null) button.removeAttribute("title");
    tooltip.textContent = label;
    tooltip.showPopover();
    const bounds = button.getBoundingClientRect();
    const tip = tooltip.getBoundingClientRect();
    const left = Math.max(8, Math.min(bounds.left + bounds.width / 2 - tip.width / 2, window.innerWidth - tip.width - 8));
    const below = bounds.bottom + tip.height + 8 <= window.innerHeight;
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${below ? bounds.bottom + 8 : Math.max(8, bounds.top - tip.height - 8)}px`;
  }

  root.addEventListener("pointerover", (event) => {
    const button = event.target.closest?.("button");
    if (button) show(button);
  });
  root.addEventListener("pointerout", (event) => {
    if (active && !active.contains(event.relatedTarget)) hide();
  });
  root.addEventListener("focusin", (event) => {
    const button = event.target.closest?.("button");
    if (button) show(button);
  });
  root.addEventListener("focusout", (event) => {
    if (active === event.target) hide();
  });
  root.addEventListener("scroll", hide, true);
  window.addEventListener("resize", hide);
}

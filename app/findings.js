// Findings contents list: mark the section being read. Clicking a link is a
// plain anchor (the browser scrolls and sets the hash; smooth unless the
// reader prefers reduced motion, see findings.css and site.css).
(function () {
  const links = [...document.querySelectorAll(".toc a")];
  const heads = links.map(a => document.getElementById(a.hash.slice(1)));
  let queued = false;
  function mark() {
    queued = false;
    const line = innerHeight * 0.3;
    let i = 0;
    heads.forEach((h, j) => { if (h.getBoundingClientRect().top <= line) i = j; });
    if (innerHeight + scrollY >= document.documentElement.scrollHeight - 2) i = heads.length - 1;   // bottom of the page
    links.forEach((a, j) => j === i ? a.setAttribute("aria-current", "location") : a.removeAttribute("aria-current"));
  }
  addEventListener("scroll", () => { if (!queued) { queued = true; requestAnimationFrame(mark); } }, { passive: true });
  addEventListener("resize", mark);
  addEventListener("hashchange", mark);
  mark();
})();

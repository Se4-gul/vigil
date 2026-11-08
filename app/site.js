// Site-wide navigation. One list of pages; unbuilt pages show a one-line
// placeholder. Set SHOW_UNFINISHED to false before publishing: unbuilt pages
// disappear from the nav, and visiting one redirects to the replay.
const SHOW_UNFINISHED = false;

const PAGES = [
  { href: "index.html",     title: "Replay",    built: true },
  { href: "compare.html",   title: "Compare",   built: true },
  { href: "whatif.html",    title: "What-if",   built: false },
  { href: "landscape.html", title: "Landscape", built: false },
  { href: "findings.html",  title: "Findings",  built: false },
  { href: "about.html",     title: "About",     built: true },
];

(function () {
  const here = location.pathname.split("/").pop() || "index.html";
  const page = PAGES.find(p => p.href === here);
  if (page && !page.built && !SHOW_UNFINISHED) {
    location.replace("index.html");
    return;
  }
  const nav = document.getElementById("site-nav");
  if (!nav) return;
  const ul = document.createElement("ul");
  for (const p of PAGES) {
    if (!p.built && !SHOW_UNFINISHED) continue;
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = p.href;
    a.textContent = p.title;
    if (p.href === here) a.setAttribute("aria-current", "page");
    if (!p.built) a.classList.add("unbuilt");
    li.appendChild(a);
    ul.appendChild(li);
  }
  nav.appendChild(ul);
})();

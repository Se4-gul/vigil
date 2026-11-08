// Site-wide navigation. One list of pages; unbuilt pages show a one-line
// placeholder. Set SHOW_UNFINISHED to false before publishing: unbuilt pages
// disappear from the nav, and visiting one redirects to the home page.
// A page marked `preview` is shown on a local server only (localhost), so a
// draft can be checked locally while staying hidden on the live site.
const SHOW_UNFINISHED = false;
const LOCAL = ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);

const PAGES = [
  { href: "replay.html",    title: "Replay",    built: true },
  { href: "landscape.html", title: "Landscape", built: true },
  { href: "compare.html",   title: "Compare",   built: true },
  { href: "whatif.html",    title: "What-if",   built: false },
  { href: "findings.html",  title: "Findings",  built: true },
  { href: "about.html",     title: "About",     built: true },
];

// Hosts may serve pages without ".html" (Cloudflare Pages redirects
// /compare.html to /compare), so match on the bare name.
const bare = s => s.replace(/\.html$/, "") || "index";
const shown = p => p.built || SHOW_UNFINISHED || (p.preview && LOCAL);

(function () {
  const here = bare(location.pathname.split("/").pop());
  const page = PAGES.find(p => bare(p.href) === here);
  if (page && !shown(page)) {
    location.replace("/");
    return;
  }
  // Links elsewhere on a page that point at a hidden page are removed too
  // (e.g. the home page's link to Findings), marked data-page="findings".
  for (const a of document.querySelectorAll("[data-page]")) {
    const p = PAGES.find(q => bare(q.href) === a.dataset.page);
    if (p && !shown(p)) a.remove();
  }
  const nav = document.getElementById("site-nav");
  if (!nav) return;
  const ul = document.createElement("ul");
  for (const p of PAGES) {
    if (!shown(p)) continue;
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = "/" + p.href;            // absolute, so links also work from the 404 page at any path
    a.textContent = p.title;
    if (bare(p.href) === here) a.setAttribute("aria-current", "page");
    if (!p.built && !p.preview) a.classList.add("unbuilt");
    li.appendChild(a);
    ul.appendChild(li);
  }
  nav.appendChild(ul);
})();

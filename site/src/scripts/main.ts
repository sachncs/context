// Scroll reveal
const revealEls = document.querySelectorAll<HTMLElement>("[data-reveal]");
if ("IntersectionObserver" in window) {
  const io = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          entry.target.classList.add("rs-visible");
          io.unobserve(entry.target);
        }
      }
    },
    { threshold: 0.12, rootMargin: "0px 0px -60px 0px" }
  );
  revealEls.forEach((el) => io.observe(el));
} else {
  revealEls.forEach((el) => el.classList.add("rs-visible"));
}

// Nav scrolled state
const nav = document.getElementById("site-nav");
const onScroll = () => nav?.classList.toggle("is-scrolled", window.scrollY > 24);
onScroll();
window.addEventListener("scroll", onScroll, { passive: true });

// Mobile menu
const menuBtn = document.querySelector<HTMLButtonElement>("[data-menu-btn]");
const mobileMenu = document.getElementById("mobile-menu");
menuBtn?.addEventListener("click", () => {
  const open = mobileMenu?.classList.toggle("is-open") ?? false;
  menuBtn.setAttribute("aria-expanded", String(open));
  document.body.style.overflow = open ? "hidden" : "";
});
mobileMenu?.querySelectorAll("a").forEach((a) =>
  a.addEventListener("click", () => {
    mobileMenu.classList.remove("is-open");
    menuBtn?.setAttribute("aria-expanded", "false");
    document.body.style.overflow = "";
  })
);

// Spotlight cards
document.querySelectorAll<HTMLElement>(".spot").forEach((card) => {
  card.addEventListener("pointermove", (e) => {
    const r = card.getBoundingClientRect();
    card.style.setProperty("--mx", `${e.clientX - r.left}px`);
    card.style.setProperty("--my", `${e.clientY - r.top}px`);
  });
});

// Count-up metrics
const countEls = document.querySelectorAll<HTMLElement>("[data-count]");
const animateCount = (el: HTMLElement) => {
  const target = Number(el.dataset.count ?? "0");
  const dur = 1500;
  const start = performance.now();
  const tick = (now: number) => {
    const p = Math.min((now - start) / dur, 1);
    const eased = 1 - Math.pow(1 - p, 4);
    el.textContent = String(Math.round(target * eased));
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
};
if ("IntersectionObserver" in window) {
  const cio = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          animateCount(e.target as HTMLElement);
          cio.unobserve(e.target);
        }
      }
    },
    { threshold: 0.6 }
  );
  countEls.forEach((el) => cio.observe(el));
} else {
  countEls.forEach(animateCount);
}

// Copy buttons
document.querySelectorAll<HTMLButtonElement>("[data-copy]").forEach((btn) => {
  const text = btn.dataset.copy ?? "";
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    btn.classList.add("copied");
    setTimeout(() => btn.classList.remove("copied"), 1800);
  });
});

// Tab groups
document.querySelectorAll<HTMLElement>("[data-tabs]").forEach((group) => {
  const btns = [...group.querySelectorAll<HTMLButtonElement>("[data-tab]")];
  const panels = [...group.querySelectorAll<HTMLElement>("[data-panel]")];
  btns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.dataset.tab;
      btns.forEach((b) => {
        b.classList.toggle("tab-active", b === btn);
        b.setAttribute("aria-selected", b === btn ? "true" : "false");
      });
      panels.forEach((p) => {
        const active = p.dataset.panel === id;
        p.classList.toggle("is-active", active);
        p.setAttribute("aria-hidden", String(!active));
      });
    });
  });
});
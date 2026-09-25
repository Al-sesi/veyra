/* Veyra — landing page interactions.
   Vanilla JS, no dependencies:
   1. Scroll-reveal via IntersectionObserver
   2. Sticky header state
   3. Guard the WhatsApp placeholder links until a real wa.me URL is set
*/
(function () {
  "use strict";

  var prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- 1. Reveal on scroll ---------- */
  var revealEls = document.querySelectorAll(".reveal");
  if (prefersReduced || !("IntersectionObserver" in window)) {
    revealEls.forEach(function (el) { el.classList.add("is-visible"); });
  } else {
    var revealObserver = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            revealObserver.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.15, rootMargin: "0px 0px -40px 0px" }
    );
    revealEls.forEach(function (el) { revealObserver.observe(el); });
  }

  /* ---------- 3. Sticky header state ---------- */
  var header = document.getElementById("siteHeader");
  function onScroll() {
    if (window.scrollY > 8) header.classList.add("scrolled");
    else header.classList.remove("scrolled");
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* ---------- 4. WhatsApp placeholder links ---------- */
  /* Links marked data-whatsapp-placeholder still point at "#". Until the real
     wa.me URL is dropped into the HTML, tell visitors instead of dead-ending. */
  document.querySelectorAll("[data-whatsapp-placeholder]").forEach(function (link) {
    link.addEventListener("click", function (event) {
      event.preventDefault();
      window.alert("Veyra on WhatsApp is launching soon — try the live demo in the meantime.");
    });
  });
})();

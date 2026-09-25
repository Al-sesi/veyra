/* Veyra — landing page interactions.
   Vanilla JS, no dependencies:
   1. Scroll-reveal via IntersectionObserver
   2. Animated stat counters
   3. Sticky header state
   4. Guard the WhatsApp placeholder links until a real wa.me URL is set
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

  /* ---------- 2. Stat counters ---------- */
  function animateCounter(el) {
    var target = parseInt(el.getAttribute("data-target"), 10);
    var prefix = el.getAttribute("data-prefix") || "";
    var suffix = el.getAttribute("data-suffix") || "";
    var duration = 1400;
    var start = null;

    function frame(now) {
      if (start === null) start = now;
      var progress = Math.min((now - start) / duration, 1);
      var eased = 1 - Math.pow(1 - progress, 3); /* ease-out cubic */
      el.textContent = prefix + Math.round(target * eased) + suffix;
      if (progress < 1) requestAnimationFrame(frame);
    }

    if (prefersReduced) {
      el.textContent = prefix + target + suffix;
    } else {
      requestAnimationFrame(frame);
    }
  }

  var counters = document.querySelectorAll(".counter");
  if ("IntersectionObserver" in window) {
    var counterObserver = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            animateCounter(entry.target);
            counterObserver.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.6 }
    );
    counters.forEach(function (el) { counterObserver.observe(el); });
  } else {
    counters.forEach(animateCounter);
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

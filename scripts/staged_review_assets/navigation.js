(() => {
  "use strict";
  const byId = id => document.getElementById(id);
  const form = byId("filters");
  const fields = Object.fromEntries(["search", "batch", "model", "result", "attention"].map(id => [id, byId(id)]));
  const rows = Array.from(document.querySelectorAll("#case-rows tr"));
  function filter() {
    const query = fields.search.value.trim().toLowerCase();
    let count = 0;
    rows.forEach(row => {
      const d = row.dataset;
      const visible = (!query || d.search.includes(query)) &&
        (!fields.batch.value || d.batch === fields.batch.value) &&
        (!fields.model.value || d.model === fields.model.value) &&
        (!fields.result.value || d.result === fields.result.value) &&
        (!fields.attention.checked || d.attention === "true");
      row.hidden = !visible;
      byId(d.case).hidden = !visible;
      if (visible) count += 1;
    });
    byId("showing").textContent = `Showing ${count} of ${rows.length} cases`;
    byId("no-results").hidden = count !== 0;
  }
  function reset() {
    Object.values(fields).forEach(field => { if (field.type === "checkbox") field.checked = false; else field.value = ""; });
  }
  function openCase(id) {
    const panel = byId(id);
    if (!panel || !panel.classList.contains("case-panel")) return;
    if (panel.hidden) { reset(); filter(); }
    panel.open = true;
    panel.scrollIntoView({block: "start"});
    panel.querySelector("summary").focus({preventScroll: true});
  }
  function openTrace(id) {
    const target = byId(id);
    if (!target) return;
    const panel = target.closest(".case-panel");
    if (!panel) return;
    if (panel.hidden) { reset(); filter(); }
    panel.open = true;
    if (target.tagName === "DETAILS") target.open = true;
    target.scrollIntoView({block: "start"});
  }
  form.hidden = false;
  form.addEventListener("submit", event => event.preventDefault());
  form.addEventListener("input", filter);
  form.addEventListener("change", filter);
  form.addEventListener("reset", event => { event.preventDefault(); reset(); filter(); });
  byId("failures").addEventListener("click", () => { reset(); fields.result.value = "fail"; filter(); });
  byId("flagged").addEventListener("click", () => { reset(); fields.result.value = "pass"; fields.attention.checked = true; filter(); });
  document.querySelectorAll("[data-pick-batch]").forEach(button => button.addEventListener("click", () => {
    reset(); fields.batch.value = button.dataset.pickBatch; fields.model.value = button.dataset.pickModel;
    filter(); byId("case-index").scrollIntoView({block: "start"});
  }));
  document.querySelectorAll("[data-open-case]").forEach(link => link.addEventListener("click", () => openCase(link.hash.slice(1))));
  document.querySelectorAll("[data-open-trace]").forEach(link => link.addEventListener("click", () => openTrace(link.hash.slice(1))));
  window.addEventListener("hashchange", () => { openCase(location.hash.slice(1)); openTrace(location.hash.slice(1)); });
  filter();
  openCase(location.hash.slice(1));
  openTrace(location.hash.slice(1));
})();

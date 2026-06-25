// ===========================================================================
// script.js — chart rendering + prediction form wiring for the dashboard
// ===========================================================================

(function () {
  const statsEl = document.getElementById("stats-data");
  const stats = JSON.parse(statsEl.textContent);

  // ---- Pie chart: Approved vs Rejected -----------------------------------
  try {
    const ctx = document.getElementById("outcomeChart").getContext("2d");
    new Chart(ctx, {
      type: "pie",
      data: {
        labels: ["Approved", "Rejected"],
        datasets: [
          {
            data: [stats.approved_loans, stats.rejected_loans],
            backgroundColor: ["#2F6F4E", "#B23A2F"],
            borderColor: "#ffffff",
            borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (item) => {
                const total = stats.approved_loans + stats.rejected_loans;
                const pct = ((item.parsed / total) * 100).toFixed(2);
                return `${item.label}: ${item.parsed} (${pct}%)`;
              },
            },
          },
        },
      },
    });
  } catch (err) {
    console.error("Chart failed to render:", err);
  }

  // ---- Prediction form -----------------------------------------------------
  const form = document.getElementById("predict-form");
  const btn = document.getElementById("predict-btn");
  const btnText = btn.querySelector(".btn-text");
  const btnSpinner = btn.querySelector(".btn-spinner");

  const placeholder = document.getElementById("result-placeholder");
  const stampWrap = document.getElementById("stamp-wrap");
  const stamp = document.getElementById("stamp");
  const stampText = document.getElementById("stamp-text");
  const stampSub = document.getElementById("stamp-sub");
  const headline = document.getElementById("result-headline");
  const meta = document.getElementById("result-meta");
  const confidenceEl = document.getElementById("result-confidence");
  const confidenceFill = document.getElementById("confidence-fill");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    btn.disabled = true;
    btnText.textContent = "Analyzing…";
    btnSpinner.classList.remove("d-none");

    const formData = new FormData(form);
    const payload = Object.fromEntries(formData.entries());

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Prediction failed");
      }

      renderResult(data);
    } catch (err) {
      renderError(err.message);
    } finally {
      btn.disabled = false;
      btnText.textContent = "Predict Loan Status";
      btnSpinner.classList.add("d-none");
    }
  });

  function renderResult(data) {
    placeholder.classList.add("d-none");
    stampWrap.classList.remove("d-none");

    stamp.classList.remove("stamp-in");
    // restart animation
    void stamp.offsetWidth;

    if (data.approved) {
      stamp.classList.remove("is-rejected");
      stampText.textContent = "Approved";
      stampSub.textContent = "Random Forest Decision";
      headline.innerHTML = 'Loan Approved <span aria-hidden="true">✅</span>';
      confidenceFill.style.background = "var(--ledger-green)";
    } else {
      stamp.classList.add("is-rejected");
      stampText.textContent = "Rejected";
      stampSub.textContent = "Random Forest Decision";
      headline.innerHTML = 'Loan Rejected <span aria-hidden="true">❌</span>';
      confidenceFill.style.background = "var(--stamp-red)";
    }

    meta.innerHTML = `Model confidence: <strong>${data.confidence}%</strong> &middot; P(approve) ${data.probability_approved}% / P(reject) ${data.probability_rejected}%`;
    confidenceEl.textContent = `${data.confidence}%`;
    confidenceFill.style.width = "0%";
    requestAnimationFrame(() => {
      confidenceFill.style.width = `${data.confidence}%`;
    });

    stamp.classList.add("stamp-in");
  }

  function renderError(message) {
    placeholder.classList.remove("d-none");
    stampWrap.classList.add("d-none");
    placeholder.textContent = `Could not generate a prediction: ${message}`;
  }
})();

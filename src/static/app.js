const form = document.querySelector("#review-form");
const reviewInput = document.querySelector("#review");
const characterCount = document.querySelector("#character-count");
const submitButton = document.querySelector("#submit-button");
const resultPanel = document.querySelector("#result-panel");
const resultLabel = document.querySelector("#result-label");
const resultValue = document.querySelector("#result-value");
const resultMessage = document.querySelector("#result-message");
const confidenceBlock = document.querySelector("#confidence-block");
const confidenceValue = document.querySelector("#confidence-value");
const confidenceTrack = document.querySelector("#confidence-track");
const confidenceFill = document.querySelector("#confidence-fill");

function updateCharacterCount() {
  characterCount.textContent = reviewInput.value.length.toLocaleString();
}

function setResult({ state, label, value, message, confidence = null }) {
  resultPanel.dataset.state = state;
  resultLabel.textContent = label;
  resultValue.textContent = value;
  resultMessage.textContent = message;
  const hasConfidence = confidence !== null;
  confidenceBlock.hidden = !hasConfidence;

  if (hasConfidence) {
    const percentage = Math.round(confidence * 1000) / 10;
    confidenceValue.textContent = `${percentage}%`;
    confidenceTrack.setAttribute("aria-valuenow", String(percentage));
    confidenceFill.style.width = `${percentage}%`;
  } else {
    confidenceFill.style.width = "0";
  }
}

function setBusy(isBusy) {
  submitButton.disabled = isBusy;
  submitButton.textContent = isBusy ? "Analyzing..." : "Analyze review";
}

reviewInput.addEventListener("input", updateCharacterCount);

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    reviewInput.value = button.dataset.example;
    updateCharacterCount();
    reviewInput.focus();
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = reviewInput.value.trim();

  if (!text) {
    setResult({ state: "error", label: "Input required", value: "No review", message: "Enter a movie review before requesting a prediction." });
    reviewInput.focus();
    return;
  }

  setBusy(true);
  setResult({ state: "loading", label: "Processing", value: "Analyzing", message: "Running the review through the sentiment model." });

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const result = await response.json();

    if (!response.ok) {
      throw new Error("The prediction service could not process this review.");
    }

    setResult({ state: result.sentiment, label: "Predicted sentiment", value: result.sentiment, message: "Prediction completed successfully.", confidence: result.confidence });
  } catch (error) {
    setResult({ state: "error", label: "Request failed", value: "Unavailable", message: error.message });
  } finally {
    setBusy(false);
  }
});
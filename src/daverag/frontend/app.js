const form = document.getElementById("chat-form");
const questionInput = document.getElementById("question");
const topicInput = document.getElementById("topic");
const answerText = document.getElementById("answer-text");
const requestState = document.getElementById("request-state");
const confidenceValue = document.getElementById("confidence");
const insufficientContextValue = document.getElementById("insufficient-context");
const sourceCountValue = document.getElementById("source-count");
const sourcesList = document.getElementById("sources");
const traceStrip = document.getElementById("trace-strip");

const TRACE_STEPS = [
  "Received question",
  "Embedding query",
  "Searching chunks",
  "Reranking results",
  "Generating answer",
];

function renderTrace(activeIndex) {
  traceStrip.innerHTML = "";
  TRACE_STEPS.forEach((step, index) => {
    const node = document.createElement("div");
    node.className = index <= activeIndex ? "trace-step active" : "trace-step";
    node.textContent = step;
    traceStrip.appendChild(node);
  });
}

function formatBool(value) {
  return value ? "true" : "false";
}

function setLoadingState(isLoading) {
  requestState.textContent = isLoading ? "Working" : "Idle";
}

async function typeAnswer(text) {
  answerText.classList.add("typing");
  answerText.textContent = "";
  for (let i = 0; i < text.length; i += 1) {
    answerText.textContent += text[i];
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  answerText.classList.remove("typing");
}

function renderSources(sources) {
  if (!sources.length) {
    sourcesList.className = "sources-list empty-state";
    sourcesList.textContent = "No sources used.";
    return;
  }
  sourcesList.className = "sources-list";
  sourcesList.innerHTML = "";
  sources.forEach((source) => {
    const card = document.createElement("article");
    card.className = "source-card";
    card.innerHTML = `
      <h4>${source.title}</h4>
      <div class="source-meta">
        <span class="chip">source_id: ${source.source_id}</span>
        <span class="chip">topic: ${source.topic}</span>
      </div>
    `;
    sourcesList.appendChild(card);
  });
}

async function askQuestion(question, topic) {
  setLoadingState(true);
  renderTrace(0);
  confidenceValue.textContent = "-";
  insufficientContextValue.textContent = "-";
  sourceCountValue.textContent = "0";

  try {
    renderTrace(1);
    renderTrace(2);
    const response = await fetch("/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        topic: topic || null,
      }),
    });

    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    renderTrace(3);
    const data = await response.json();
    renderTrace(4);

    confidenceValue.textContent = data.confidence;
    insufficientContextValue.textContent = formatBool(data.insufficient_context);
    sourceCountValue.textContent = String(data.sources.length);

    renderSources(data.sources);
    await typeAnswer(data.answer);
  } catch (error) {
    answerText.classList.remove("typing");
    answerText.textContent = error.message;
    renderSources([]);
  } finally {
    setLoadingState(false);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  const topic = topicInput.value.trim();
  if (!question) {
    return;
  }
  await askQuestion(question, topic);
});

renderTrace(-1);
